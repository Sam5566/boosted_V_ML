ЦЪ
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ль
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	А*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
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
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
Г
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А* 
shared_nameconv2d_7/kernel
|
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*'
_output_shapes
: А*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:А*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:А*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
З
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_8/kernel/m
А
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	А*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_6/kernel/m
Й
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
: *
dtype0
А
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
: *
dtype0
С
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*'
shared_nameAdam/conv2d_7/kernel/m
К
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*'
_output_shapes
: А*
dtype0
Б
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_7/bias/m
z
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_8/kernel/m
Л
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_8/bias/m
z
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_6/kernel/m
Б
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
АА*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_7/kernel/m
Б
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m* 
_output_shapes
:
АА*
dtype0

Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_7/bias/m
x
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_8/kernel/v
А
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	А*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_6/kernel/v
Й
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
: *
dtype0
А
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
: *
dtype0
С
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*'
shared_nameAdam/conv2d_7/kernel/v
К
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*'
_output_shapes
: А*
dtype0
Б
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_7/bias/v
z
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_8/kernel/v
Л
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_8/bias/v
z
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes	
:А*
dtype0
И
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_6/kernel/v
Б
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
АА*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:А*
dtype0
И
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_7/kernel/v
Б
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v* 
_output_shapes
:
АА*
dtype0

Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_7/bias/v
x
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
ƒ]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*€\
valueх\Bт\ Bл\
К

h2ptjl
_output
	optimizer
	variables
regularization_losses
trainable_variables
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
Ў
!iter

"beta_1

#beta_2
	$decay
%learning_ratemЌmќ&mѕ'm–*m—+m“,m”-m‘.m’/m÷0m„1mЎ2mў3mЏvџv№&vЁ'vё*vя+vа,vб-vв.vг/vд0vе1vж2vз3vи
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
≠
4non_trainable_variables
5layer_regularization_losses

6layers
	variables
regularization_losses
7layer_metrics
8metrics
trainable_variables
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
≠
rnon_trainable_variables
slayer_regularization_losses

tlayers
	variables
regularization_losses
ulayer_metrics
vmetrics
trainable_variables
MK
VARIABLE_VALUEdense_8/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_8/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
wnon_trainable_variables
xlayer_regularization_losses

ylayers
	variables
regularization_losses
zlayer_metrics
{metrics
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
VARIABLE_VALUEbatch_normalization_2/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_2/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_2/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_2/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_6/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_6/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_7/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_7/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_8/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_8/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_6/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_6/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_7/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_7/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

0
1
 

|0
}1
 
 
 
∞
~non_trainable_variables
layer_regularization_losses
Аlayers
9	variables
:regularization_losses
Бlayer_metrics
Вmetrics
;trainable_variables
 

&0
'1
(2
)3
 

&0
'1
≤
Гnon_trainable_variables
 Дlayer_regularization_losses
Еlayers
>	variables
?regularization_losses
Жlayer_metrics
Зmetrics
@trainable_variables

*0
+1
 

*0
+1
≤
Иnon_trainable_variables
 Йlayer_regularization_losses
Кlayers
B	variables
Cregularization_losses
Лlayer_metrics
Мmetrics
Dtrainable_variables
 
 
 
≤
Нnon_trainable_variables
 Оlayer_regularization_losses
Пlayers
F	variables
Gregularization_losses
Рlayer_metrics
Сmetrics
Htrainable_variables

,0
-1
 

,0
-1
≤
Тnon_trainable_variables
 Уlayer_regularization_losses
Фlayers
J	variables
Kregularization_losses
Хlayer_metrics
Цmetrics
Ltrainable_variables
 
 
 
≤
Чnon_trainable_variables
 Шlayer_regularization_losses
Щlayers
N	variables
Oregularization_losses
Ъlayer_metrics
Ыmetrics
Ptrainable_variables

.0
/1
 

.0
/1
≤
Ьnon_trainable_variables
 Эlayer_regularization_losses
Юlayers
R	variables
Sregularization_losses
Яlayer_metrics
†metrics
Ttrainable_variables
 
 
 
≤
°non_trainable_variables
 Ґlayer_regularization_losses
£layers
V	variables
Wregularization_losses
§layer_metrics
•metrics
Xtrainable_variables
 
 
 
≤
¶non_trainable_variables
 Іlayer_regularization_losses
®layers
Z	variables
[regularization_losses
©layer_metrics
™metrics
\trainable_variables
 
 
 
≤
Ђnon_trainable_variables
 ђlayer_regularization_losses
≠layers
^	variables
_regularization_losses
Ѓlayer_metrics
ѓmetrics
`trainable_variables

00
11
 

00
11
≤
∞non_trainable_variables
 ±layer_regularization_losses
≤layers
b	variables
cregularization_losses
≥layer_metrics
іmetrics
dtrainable_variables
 
 
 
≤
µnon_trainable_variables
 ґlayer_regularization_losses
Јlayers
f	variables
gregularization_losses
Єlayer_metrics
єmetrics
htrainable_variables

20
31
 

20
31
≤
Їnon_trainable_variables
 їlayer_regularization_losses
Љlayers
j	variables
kregularization_losses
љlayer_metrics
Њmetrics
ltrainable_variables
 
 
 
≤
њnon_trainable_variables
 јlayer_regularization_losses
Ѕlayers
n	variables
oregularization_losses
¬layer_metrics
√metrics
ptrainable_variables

(0
)1
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
pn
VARIABLE_VALUEAdam/dense_8/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_8/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_6/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_6/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_7/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_7/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_8/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_8/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_6/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_6/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_7/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_7/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dense_8/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_8/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_6/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_6/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_7/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_7/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_8/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_8/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_6/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_6/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_7/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_7/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
Ф
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias*
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
GPU2 *0J 8В *-
f(R&
$__inference_signature_wrapper_391258
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
≈
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*B
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
__inference__traced_save_392945
Ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biastotalcounttotal_1count_1Adam/dense_8/kernel/mAdam/dense_8/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*A
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
"__inference__traced_restore_393114ЕР
і
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_390340

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
¶
—
6__inference_batch_normalization_2_layer_call_fn_392489

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3901252
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
ј
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392414

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
„
F
*__inference_flatten_2_layer_call_fn_392612

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3902082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
й
Д
-__inference_sequential_2_layer_call_fn_392332
lambda_2_input
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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3906072
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_namelambda_2_input
Ц
ђ
$__inference_CNN_layer_call_fn_391771

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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCall≠
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
GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_3910112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–
±
__inference_loss_fn_2_392763M
9dense_7_kernel_regularizer_square_readvariableop_resource:
АА
identityИҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpа
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_7_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulШ
IdentityIdentity"dense_7/kernel/Regularizer/mul:z:01^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp
¬g
£
__inference__traced_save_392945
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
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
ShardedFilenameж
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*ш
valueоBл6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices“
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*з
_input_shapes’
“: :	А:: : : : : ::::: : : А:А:АА:А:
АА:А:
АА:А: : : : :	А:::: : : А:А:АА:А:
АА:А:
АА:А:	А:::: : : А:А:АА:А:
АА:А:
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
:А:&"
 
_output_shapes
:
АА:!
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
:А:&$"
 
_output_shapes
:
АА:!%
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
:А:&2"
 
_output_shapes
:
АА:!3
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
№
L
0__inference_max_pooling2d_7_layer_call_fn_390079

inputs
identityс
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
GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3900732
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
Ѕt
О
H__inference_sequential_2_layer_call_and_return_conditional_losses_392096
lambda_2_input;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: B
'conv2d_7_conv2d_readvariableop_resource: А7
(conv2d_7_biasadd_readvariableop_resource:	АC
'conv2d_8_conv2d_readvariableop_resource:АА7
(conv2d_8_biasadd_readvariableop_resource:	А:
&dense_6_matmul_readvariableop_resource:
АА6
'dense_6_biasadd_readvariableop_resource:	А:
&dense_7_matmul_readvariableop_resource:
АА6
'dense_7_biasadd_readvariableop_resource:	А
identityИҐ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpХ
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_2/strided_slice/stackЩ
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_2/strided_slice/stack_1Щ
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_2/strided_slice/stack_2≤
lambda_2/strided_sliceStridedSlicelambda_2_input%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2
lambda_2/strided_sliceґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1з
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3lambda_2/strided_slice:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOpв
conv2d_6/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/Relu«
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool±
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02 
conv2d_7/Conv2D/ReadVariableOpў
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_7/Conv2D®
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp≠
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/BiasAdd|
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/Relu»
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool≤
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_8/Conv2D/ReadVariableOpў
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_8/Conv2D®
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp≠
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/Relu»
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPoolС
dropout_6/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_6/Identitys
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
flatten_2/ConstЫ
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_2/ReshapeІ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_6/MatMul/ReadVariableOp†
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/MatMul•
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_6/BiasAdd/ReadVariableOpҐ
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/ReluГ
dropout_7/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/IdentityІ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_7/MatMul/ReadVariableOp°
dense_7/MatMulMatMuldropout_7/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/MatMul•
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_7/BiasAdd/ReadVariableOpҐ
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/ReluГ
dropout_8/IdentityIdentitydense_7/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_8/Identity÷
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЌ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЌ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulЧ
IdentityIdentitydropout_8/Identity:output:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_namelambda_2_input
•
Ш
(__inference_dense_6_layer_call_fn_392644

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallщ
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3902272
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ч
ј
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392450

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
7:€€€€€€€€€:::::*
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
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ВУ
Ь
?__inference_CNN_layer_call_and_return_conditional_losses_391348

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2л
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpЦ
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpа
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/BiasAddҐ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/Reluо
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpН
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpб
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/BiasAdd£
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/Reluп
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpН
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpб
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/BiasAdd£
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/Reluп
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolЄ
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_6/IdentityН
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Constѕ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOp‘
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp÷
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/BiasAddШ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/Relu™
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_7/Identityќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp’
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp÷
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/BiasAddШ
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/Relu™
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_8/Identity¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp≠
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp°
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/BiasAddy
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/Softmaxг
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЏ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЏ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulЛ
IdentityIdentitydense_8/Softmax:softmax:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
щё
•!
"__inference__traced_restore_393114
file_prefix2
assignvariableop_dense_8_kernel:	А-
assignvariableop_1_dense_8_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_2_gamma:;
-assignvariableop_8_batch_normalization_2_beta:B
4assignvariableop_9_batch_normalization_2_moving_mean:G
9assignvariableop_10_batch_normalization_2_moving_variance:=
#assignvariableop_11_conv2d_6_kernel: /
!assignvariableop_12_conv2d_6_bias: >
#assignvariableop_13_conv2d_7_kernel: А0
!assignvariableop_14_conv2d_7_bias:	А?
#assignvariableop_15_conv2d_8_kernel:АА0
!assignvariableop_16_conv2d_8_bias:	А6
"assignvariableop_17_dense_6_kernel:
АА/
 assignvariableop_18_dense_6_bias:	А6
"assignvariableop_19_dense_7_kernel:
АА/
 assignvariableop_20_dense_7_bias:	А#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: <
)assignvariableop_25_adam_dense_8_kernel_m:	А5
'assignvariableop_26_adam_dense_8_bias_m:D
6assignvariableop_27_adam_batch_normalization_2_gamma_m:C
5assignvariableop_28_adam_batch_normalization_2_beta_m:D
*assignvariableop_29_adam_conv2d_6_kernel_m: 6
(assignvariableop_30_adam_conv2d_6_bias_m: E
*assignvariableop_31_adam_conv2d_7_kernel_m: А7
(assignvariableop_32_adam_conv2d_7_bias_m:	АF
*assignvariableop_33_adam_conv2d_8_kernel_m:АА7
(assignvariableop_34_adam_conv2d_8_bias_m:	А=
)assignvariableop_35_adam_dense_6_kernel_m:
АА6
'assignvariableop_36_adam_dense_6_bias_m:	А=
)assignvariableop_37_adam_dense_7_kernel_m:
АА6
'assignvariableop_38_adam_dense_7_bias_m:	А<
)assignvariableop_39_adam_dense_8_kernel_v:	А5
'assignvariableop_40_adam_dense_8_bias_v:D
6assignvariableop_41_adam_batch_normalization_2_gamma_v:C
5assignvariableop_42_adam_batch_normalization_2_beta_v:D
*assignvariableop_43_adam_conv2d_6_kernel_v: 6
(assignvariableop_44_adam_conv2d_6_bias_v: E
*assignvariableop_45_adam_conv2d_7_kernel_v: А7
(assignvariableop_46_adam_conv2d_7_bias_v:	АF
*assignvariableop_47_adam_conv2d_8_kernel_v:АА7
(assignvariableop_48_adam_conv2d_8_bias_v:	А=
)assignvariableop_49_adam_dense_6_kernel_v:
АА6
'assignvariableop_50_adam_dense_6_bias_v:	А=
)assignvariableop_51_adam_dense_7_kernel_v:
АА6
'assignvariableop_52_adam_dense_7_bias_v:	А
identity_54ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9м
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*ш
valueоBл6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_2_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≤
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_2_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9є
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_2_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ѕ
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_2_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ђ
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_6_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_6_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ђ
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_7_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_7_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ђ
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_8_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16©
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv2d_8_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17™
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_6_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_6_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19™
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_7_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_7_biasIdentity_20:output:0"/device:CPU:0*
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
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_8_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ѓ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_8_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Њ
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_2_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28љ
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_2_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≤
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_6_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30∞
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_6_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≤
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_7_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32∞
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_7_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33≤
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_8_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34∞
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_8_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35±
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_6_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ѓ
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_6_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_7_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ѓ
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_7_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_8_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ѓ
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_8_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Њ
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_2_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42љ
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_2_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≤
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_6_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44∞
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_6_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45≤
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_7_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46∞
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_7_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47≤
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_8_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48∞
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_8_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49±
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_6_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ѓ
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_6_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51±
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_7_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52ѓ
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_7_bias_vIdentity_52:output:0"/device:CPU:0*
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
£
™
C__inference_dense_6_layer_call_and_return_conditional_losses_392635

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
Relu≈
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЋ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ц
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_390200

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¬
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_392368

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
:€€€€€€€€€*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
©
±
D__inference_conv2d_6_layer_call_and_return_conditional_losses_390152

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЌ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul”
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•1
£
?__inference_CNN_layer_call_and_return_conditional_losses_391011

inputs!
sequential_2_390958:!
sequential_2_390960:!
sequential_2_390962:!
sequential_2_390964:-
sequential_2_390966: !
sequential_2_390968: .
sequential_2_390970: А"
sequential_2_390972:	А/
sequential_2_390974:АА"
sequential_2_390976:	А'
sequential_2_390978:
АА"
sequential_2_390980:	А'
sequential_2_390982:
АА"
sequential_2_390984:	А!
dense_8_390987:	А
dense_8_390989:
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐdense_8/StatefulPartitionedCallҐ$sequential_2/StatefulPartitionedCallј
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_390958sequential_2_390960sequential_2_390962sequential_2_390964sequential_2_390966sequential_2_390968sequential_2_390970sequential_2_390972sequential_2_390974sequential_2_390976sequential_2_390978sequential_2_390980sequential_2_390982sequential_2_390984*
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3906072&
$sequential_2/StatefulPartitionedCallї
dense_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0dense_8_390987dense_8_390989*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3908462!
dense_8/StatefulPartitionedCall¬
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_390966*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЇ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_390978* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЇ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_390982* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulя
IdentityIdentity(dense_8/StatefulPartitionedCall:output:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp ^dense_8/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
†
А
D__inference_conv2d_8_layer_call_and_return_conditional_losses_392565

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
:€€€€€€€€€А*
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
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¬
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_390505

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
:€€€€€€€€€*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–
±
__inference_loss_fn_1_392752M
9dense_6_kernel_regularizer_square_readvariableop_resource:
АА
identityИҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpа
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_6_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulШ
IdentityIdentity"dense_6/kernel/Regularizer/mul:z:01^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp
¬
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_392360

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
:€€€€€€€€€*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ј

х
C__inference_dense_8_layer_call_and_return_conditional_losses_390846

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
£
™
C__inference_dense_7_layer_call_and_return_conditional_losses_390257

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
Relu≈
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulЋ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
х
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_390412

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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
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
:€€€€€€€€€А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
”
ь
-__inference_sequential_2_layer_call_fn_392266

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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИҐStatefulPartitionedCallЭ
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3902892
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
љr
Џ
__inference_call_306610

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2г
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:А*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ї
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:А:::::*
epsilon%oГ:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpО
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpЎ
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:А 2
sequential_2/conv2d_6/BiasAddЪ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:А 2
sequential_2/conv2d_6/Reluж
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*'
_output_shapes
:А *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpЕ
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpў
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_7/BiasAddЫ
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_7/Reluз
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpЕ
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpў
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_8/BiasAddЫ
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_8/Reluз
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPool∞
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*(
_output_shapes
:АА2!
sequential_2/dropout_6/IdentityН
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Const«
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0* 
_output_shapes
:
АА2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOpћ
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOpќ
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_6/BiasAddР
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_6/ReluҐ
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0* 
_output_shapes
:
АА2!
sequential_2/dropout_7/Identityќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOpЌ
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOpќ
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_7/BiasAddР
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_7/ReluҐ
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0* 
_output_shapes
:
АА2!
sequential_2/dropout_8/Identity¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp•
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOpЩ
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_8/BiasAddq
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_8/Softmaxй
IdentityIdentitydense_8/Softmax:softmax:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:А: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
єњ
Ж
?__inference_CNN_layer_call_and_return_conditional_losses_391459

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐ1sequential_2/batch_normalization_2/AssignNewValueҐ3sequential_2/batch_normalization_2/AssignNewValue_1ҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2л
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1–
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<25
3sequential_2/batch_normalization_2/FusedBatchNormV3с
1sequential_2/batch_normalization_2/AssignNewValueAssignVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource@sequential_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_2/batch_normalization_2/AssignNewValueэ
3sequential_2/batch_normalization_2/AssignNewValue_1AssignVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceDsequential_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0E^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_2/batch_normalization_2/AssignNewValue_1„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpЦ
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpа
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/BiasAddҐ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/Reluо
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpН
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpб
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/BiasAdd£
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/Reluп
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpН
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpб
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/BiasAdd£
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/Reluп
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolС
$sequential_2/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2&
$sequential_2/dropout_6/dropout/Constи
"sequential_2/dropout_6/dropout/MulMul-sequential_2/max_pooling2d_8/MaxPool:output:0-sequential_2/dropout_6/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2$
"sequential_2/dropout_6/dropout/Mul©
$sequential_2/dropout_6/dropout/ShapeShape-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_6/dropout/ShapeВ
;sequential_2/dropout_6/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_6/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02=
;sequential_2/dropout_6/dropout/random_uniform/RandomUniform£
-sequential_2/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-sequential_2/dropout_6/dropout/GreaterEqual/y£
+sequential_2/dropout_6/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_6/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_6/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2-
+sequential_2/dropout_6/dropout/GreaterEqualЌ
#sequential_2/dropout_6/dropout/CastCast/sequential_2/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2%
#sequential_2/dropout_6/dropout/Castя
$sequential_2/dropout_6/dropout/Mul_1Mul&sequential_2/dropout_6/dropout/Mul:z:0'sequential_2/dropout_6/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_2/dropout_6/dropout/Mul_1Н
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Constѕ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/dropout/Mul_1:z:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOp‘
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp÷
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/BiasAddШ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/ReluС
$sequential_2/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_2/dropout_7/dropout/ConstЏ
"sequential_2/dropout_7/dropout/MulMul'sequential_2/dense_6/Relu:activations:0-sequential_2/dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"sequential_2/dropout_7/dropout/Mul£
$sequential_2/dropout_7/dropout/ShapeShape'sequential_2/dense_6/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_7/dropout/Shapeъ
;sequential_2/dropout_7/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02=
;sequential_2/dropout_7/dropout/random_uniform/RandomUniform£
-sequential_2/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_2/dropout_7/dropout/GreaterEqual/yЫ
+sequential_2/dropout_7/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_7/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2-
+sequential_2/dropout_7/dropout/GreaterEqual≈
#sequential_2/dropout_7/dropout/CastCast/sequential_2/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2%
#sequential_2/dropout_7/dropout/Cast„
$sequential_2/dropout_7/dropout/Mul_1Mul&sequential_2/dropout_7/dropout/Mul:z:0'sequential_2/dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2&
$sequential_2/dropout_7/dropout/Mul_1ќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp’
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/dropout/Mul_1:z:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp÷
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/BiasAddШ
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/ReluС
$sequential_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_2/dropout_8/dropout/ConstЏ
"sequential_2/dropout_8/dropout/MulMul'sequential_2/dense_7/Relu:activations:0-sequential_2/dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"sequential_2/dropout_8/dropout/Mul£
$sequential_2/dropout_8/dropout/ShapeShape'sequential_2/dense_7/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_8/dropout/Shapeъ
;sequential_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02=
;sequential_2/dropout_8/dropout/random_uniform/RandomUniform£
-sequential_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_2/dropout_8/dropout/GreaterEqual/yЫ
+sequential_2/dropout_8/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_8/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2-
+sequential_2/dropout_8/dropout/GreaterEqual≈
#sequential_2/dropout_8/dropout/CastCast/sequential_2/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2%
#sequential_2/dropout_8/dropout/Cast„
$sequential_2/dropout_8/dropout/Mul_1Mul&sequential_2/dropout_8/dropout/Mul:z:0'sequential_2/dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2&
$sequential_2/dropout_8/dropout/Mul_1¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp≠
dense_8/MatMulMatMul(sequential_2/dropout_8/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp°
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/BiasAddy
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/Softmaxг
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЏ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЏ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulх
IdentityIdentitydense_8/Softmax:softmax:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp2^sequential_2/batch_normalization_2/AssignNewValue4^sequential_2/batch_normalization_2/AssignNewValue_1C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2f
1sequential_2/batch_normalization_2/AssignNewValue1sequential_2/batch_normalization_2/AssignNewValue2j
3sequential_2/batch_normalization_2/AssignNewValue_13sequential_2/batch_normalization_2/AssignNewValue_12И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
і
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_390373

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
ц
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_392649

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
њ
ј
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_389995

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
х
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_392591

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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
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
:€€€€€€€€€А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ј

х
C__inference_dense_8_layer_call_and_return_conditional_losses_392343

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
•
Ш
(__inference_dense_7_layer_call_fn_392703

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallщ
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3902572
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

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
”
c
*__inference_dropout_8_layer_call_fn_392730

inputs
identityИҐStatefulPartitionedCallб
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3903402
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
«
F
*__inference_dropout_7_layer_call_fn_392666

inputs
identity…
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_3902382
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
√
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_390125

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
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
б
E
)__inference_lambda_2_layer_call_fn_392378

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_3905052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_390085

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
»
Ю
)__inference_conv2d_6_layer_call_fn_392534

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_3901522
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь
€
D__inference_conv2d_7_layer_call_and_return_conditional_losses_390170

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
:€€€€€€€€€А*
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
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
§
—
6__inference_batch_normalization_2_layer_call_fn_392502

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3904782
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ц
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_392708

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
і
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_392720

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
ѓШ
ё
H__inference_sequential_2_layer_call_and_return_conditional_losses_392200
lambda_2_input;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: B
'conv2d_7_conv2d_readvariableop_resource: А7
(conv2d_7_biasadd_readvariableop_resource:	АC
'conv2d_8_conv2d_readvariableop_resource:АА7
(conv2d_8_biasadd_readvariableop_resource:	А:
&dense_6_matmul_readvariableop_resource:
АА6
'dense_6_biasadd_readvariableop_resource:	А:
&dense_7_matmul_readvariableop_resource:
АА6
'dense_7_biasadd_readvariableop_resource:	А
identityИҐ$batch_normalization_2/AssignNewValueҐ&batch_normalization_2/AssignNewValue_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpХ
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_2/strided_slice/stackЩ
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_2/strided_slice/stack_1Щ
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_2/strided_slice/stack_2≤
lambda_2/strided_sliceStridedSlicelambda_2_input%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2
lambda_2/strided_sliceґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3lambda_2/strided_slice:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2(
&batch_normalization_2/FusedBatchNormV3∞
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueЉ
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOpв
conv2d_6/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/Relu«
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool±
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02 
conv2d_7/Conv2D/ReadVariableOpў
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_7/Conv2D®
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp≠
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/BiasAdd|
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/Relu»
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool≤
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_8/Conv2D/ReadVariableOpў
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_8/Conv2D®
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp≠
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/Relu»
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPoolw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_6/dropout/Constі
dropout_6/dropout/MulMul max_pooling2d_8/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/MulВ
dropout_6/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shapeџ
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_6/dropout/random_uniform/RandomUniformЙ
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_6/dropout/GreaterEqual/yп
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
dropout_6/dropout/GreaterEqual¶
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/CastЂ
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/Mul_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
flatten_2/ConstЫ
flatten_2/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_2/ReshapeІ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_6/MatMul/ReadVariableOp†
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/MatMul•
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_6/BiasAdd/ReadVariableOpҐ
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/Reluw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/Const¶
dropout_7/dropout/MulMuldense_6/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Mul|
dropout_7/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape”
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_7/dropout/random_uniform/RandomUniformЙ
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/yз
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_7/dropout/GreaterEqualЮ
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Cast£
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Mul_1І
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_7/MatMul/ReadVariableOp°
dense_7/MatMulMatMuldropout_7/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/MatMul•
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_7/BiasAdd/ReadVariableOpҐ
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const¶
dropout_8/dropout/MulMuldense_7/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_8/dropout/Mul|
dropout_8/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape”
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЙ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/yз
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_8/dropout/GreaterEqualЮ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_8/dropout/Cast£
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_8/dropout/Mul_1÷
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЌ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЌ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulз
IdentityIdentitydropout_8/dropout/Mul_1:z:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_namelambda_2_input
Щ
≠
$__inference_CNN_layer_call_fn_391808
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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallЃ
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
GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_3910112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
¶
ї
!__inference__wrapped_model_389929
input_1

cnn_389895:

cnn_389897:

cnn_389899:

cnn_389901:$

cnn_389903: 

cnn_389905: %

cnn_389907: А

cnn_389909:	А&

cnn_389911:АА

cnn_389913:	А

cnn_389915:
АА

cnn_389917:	А

cnn_389919:
АА

cnn_389921:	А

cnn_389923:	А

cnn_389925:
identityИҐCNN/StatefulPartitionedCallЭ
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1
cnn_389895
cnn_389897
cnn_389899
cnn_389901
cnn_389903
cnn_389905
cnn_389907
cnn_389909
cnn_389911
cnn_389913
cnn_389915
cnn_389917
cnn_389919
cnn_389921
cnn_389923
cnn_389925*
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
GPU2 *0J 8В * 
fR
__inference_call_3046242
CNN/StatefulPartitionedCallЦ
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^CNN/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2:
CNN/StatefulPartitionedCallCNN/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
і
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_392661

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
©
±
D__inference_conv2d_6_layer_call_and_return_conditional_losses_392525

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЌ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul”
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
лZ
ч
H__inference_sequential_2_layer_call_and_return_conditional_losses_390289

inputs*
batch_normalization_2_390126:*
batch_normalization_2_390128:*
batch_normalization_2_390130:*
batch_normalization_2_390132:)
conv2d_6_390153: 
conv2d_6_390155: *
conv2d_7_390171: А
conv2d_7_390173:	А+
conv2d_8_390189:АА
conv2d_8_390191:	А"
dense_6_390228:
АА
dense_6_390230:	А"
dense_7_390258:
АА
dense_7_390260:	А
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐdense_7/StatefulPartitionedCallҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpб
lambda_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_3901062
lambda_2/PartitionedCallљ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0batch_normalization_2_390126batch_normalization_2_390128batch_normalization_2_390130batch_normalization_2_390132*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3901252/
-batch_normalization_2/StatefulPartitionedCall—
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_6_390153conv2d_6_390155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_3901522"
 conv2d_6/StatefulPartitionedCallЩ
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3900612!
max_pooling2d_6/PartitionedCallƒ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_390171conv2d_7_390173*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_3901702"
 conv2d_7/StatefulPartitionedCallЪ
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3900732!
max_pooling2d_7/PartitionedCallƒ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_390189conv2d_8_390191*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_3901882"
 conv2d_8/StatefulPartitionedCallЪ
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3900852!
max_pooling2d_8/PartitionedCallЗ
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_3902002
dropout_6/PartitionedCallщ
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3902082
flatten_2/PartitionedCall±
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_390228dense_6_390230*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3902272!
dense_6/StatefulPartitionedCall€
dropout_7/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_3902382
dropout_7/PartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_7_390258dense_7_390260*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3902572!
dense_7/StatefulPartitionedCall€
dropout_8/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3902682
dropout_8/PartitionedCallЊ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_390153*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulµ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_390228* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulµ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_390258* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulо
IdentityIdentity"dropout_8/PartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
В
є
__inference_loss_fn_0_392741T
:conv2d_6_kernel_regularizer_square_readvariableop_resource: 
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpй
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_6_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЪ
IdentityIdentity#conv2d_6/kernel/Regularizer/mul:z:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
£
™
C__inference_dense_7_layer_call_and_return_conditional_losses_392694

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
Relu≈
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulЋ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
№
L
0__inference_max_pooling2d_6_layer_call_fn_390067

inputs
identityс
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
GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3900612
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
л
Д
-__inference_sequential_2_layer_call_fn_392233
lambda_2_input
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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИҐStatefulPartitionedCall•
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3902892
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_namelambda_2_input
о
—
6__inference_batch_normalization_2_layer_call_fn_392463

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЇ
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
GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3899512
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
Ь
€
D__inference_conv2d_7_layer_call_and_return_conditional_losses_392545

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
:€€€€€€€€€А*
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
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_390073

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
ч
ј
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_390478

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
7:€€€€€€€€€:::::*
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
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
й
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_392607

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
№
L
0__inference_max_pooling2d_8_layer_call_fn_390091

inputs
identityс
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
GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3900852
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
Ђ
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_390061

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
І1
£
?__inference_CNN_layer_call_and_return_conditional_losses_390871

inputs!
sequential_2_390806:!
sequential_2_390808:!
sequential_2_390810:!
sequential_2_390812:-
sequential_2_390814: !
sequential_2_390816: .
sequential_2_390818: А"
sequential_2_390820:	А/
sequential_2_390822:АА"
sequential_2_390824:	А'
sequential_2_390826:
АА"
sequential_2_390828:	А'
sequential_2_390830:
АА"
sequential_2_390832:	А!
dense_8_390847:	А
dense_8_390849:
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐdense_8/StatefulPartitionedCallҐ$sequential_2/StatefulPartitionedCall¬
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_390806sequential_2_390808sequential_2_390810sequential_2_390812sequential_2_390814sequential_2_390816sequential_2_390818sequential_2_390820sequential_2_390822sequential_2_390824sequential_2_390826sequential_2_390828sequential_2_390830sequential_2_390832*
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3902892&
$sequential_2/StatefulPartitionedCallї
dense_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0dense_8_390847dense_8_390849*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3908462!
dense_8/StatefulPartitionedCall¬
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_390814*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЇ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_390826* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЇ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_390830* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulя
IdentityIdentity(dense_8/StatefulPartitionedCall:output:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp ^dense_8/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Л
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_389951

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
й
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_390208

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љr
Џ
__inference_call_306682

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2г
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:А*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ї
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:А:::::*
epsilon%oГ:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpО
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpЎ
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:А 2
sequential_2/conv2d_6/BiasAddЪ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:А 2
sequential_2/conv2d_6/Reluж
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*'
_output_shapes
:А *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpЕ
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpў
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_7/BiasAddЫ
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_7/Reluз
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpЕ
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpў
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_8/BiasAddЫ
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_2/conv2d_8/Reluз
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPool∞
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*(
_output_shapes
:АА2!
sequential_2/dropout_6/IdentityН
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Const«
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0* 
_output_shapes
:
АА2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOpћ
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOpќ
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_6/BiasAddР
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_6/ReluҐ
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0* 
_output_shapes
:
АА2!
sequential_2/dropout_7/Identityќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOpЌ
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOpќ
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_7/BiasAddР
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_2/dense_7/ReluҐ
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0* 
_output_shapes
:
АА2!
sequential_2/dropout_8/Identity¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp•
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOpЩ
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_8/BiasAddq
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_8/Softmaxй
IdentityIdentitydense_8/Softmax:softmax:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:А: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
э
≠
$__inference_signature_wrapper_391258
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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallТ
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
GPU2 *0J 8В **
f%R#
!__inference__wrapped_model_3899292
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
м
—
6__inference_batch_normalization_2_layer_call_fn_392476

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЄ
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
GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3899952
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
ц
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_390268

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
¬
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_390106

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
:€€€€€€€€€*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ш
ђ
$__inference_CNN_layer_call_fn_391734

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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

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
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_3908712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
†
А
D__inference_conv2d_8_layer_call_and_return_conditional_losses_390188

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
:€€€€€€€€€А*
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
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
√
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392432

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
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ц
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_390238

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
£
™
C__inference_dense_6_layer_call_and_return_conditional_losses_390227

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
Relu≈
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЋ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ѕ_
г
H__inference_sequential_2_layer_call_and_return_conditional_losses_390607

inputs*
batch_normalization_2_390547:*
batch_normalization_2_390549:*
batch_normalization_2_390551:*
batch_normalization_2_390553:)
conv2d_6_390556: 
conv2d_6_390558: *
conv2d_7_390562: А
conv2d_7_390564:	А+
conv2d_8_390568:АА
conv2d_8_390570:	А"
dense_6_390576:
АА
dense_6_390578:	А"
dense_7_390582:
АА
dense_7_390584:	А
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐdense_7/StatefulPartitionedCallҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐ!dropout_6/StatefulPartitionedCallҐ!dropout_7/StatefulPartitionedCallҐ!dropout_8/StatefulPartitionedCallб
lambda_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_3905052
lambda_2/PartitionedCallї
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0batch_normalization_2_390547batch_normalization_2_390549batch_normalization_2_390551batch_normalization_2_390553*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3904782/
-batch_normalization_2/StatefulPartitionedCall—
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_6_390556conv2d_6_390558*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_3901522"
 conv2d_6/StatefulPartitionedCallЩ
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3900612!
max_pooling2d_6/PartitionedCallƒ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_390562conv2d_7_390564*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_3901702"
 conv2d_7/StatefulPartitionedCallЪ
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3900732!
max_pooling2d_7/PartitionedCallƒ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_390568conv2d_8_390570*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_3901882"
 conv2d_8/StatefulPartitionedCallЪ
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3900852!
max_pooling2d_8/PartitionedCallЯ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_3904122#
!dropout_6/StatefulPartitionedCallБ
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3902082
flatten_2/PartitionedCall±
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_390576dense_6_390578*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3902272!
dense_6/StatefulPartitionedCallї
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_3903732#
!dropout_7/StatefulPartitionedCallє
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_7_390582dense_7_390584*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3902572!
dense_7/StatefulPartitionedCallї
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3903402#
!dropout_8/StatefulPartitionedCallЊ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_390556*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulµ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_390576* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulµ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_390582* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulв
IdentityIdentity*dropout_8/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall1^dense_7/kernel/Regularizer/Square/ReadVariableOp"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
Ц
(__inference_dense_8_layer_call_fn_392352

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCallш
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3908462
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
Л
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392396

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
©t
Ж
H__inference_sequential_2_layer_call_and_return_conditional_losses_391909

inputs;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: B
'conv2d_7_conv2d_readvariableop_resource: А7
(conv2d_7_biasadd_readvariableop_resource:	АC
'conv2d_8_conv2d_readvariableop_resource:АА7
(conv2d_8_biasadd_readvariableop_resource:	А:
&dense_6_matmul_readvariableop_resource:
АА6
'dense_6_biasadd_readvariableop_resource:	А:
&dense_7_matmul_readvariableop_resource:
АА6
'dense_7_biasadd_readvariableop_resource:	А
identityИҐ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpХ
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_2/strided_slice/stackЩ
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_2/strided_slice/stack_1Щ
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_2/strided_slice/stack_2™
lambda_2/strided_sliceStridedSliceinputs%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2
lambda_2/strided_sliceґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1з
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3lambda_2/strided_slice:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOpв
conv2d_6/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/Relu«
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool±
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02 
conv2d_7/Conv2D/ReadVariableOpў
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_7/Conv2D®
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp≠
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/BiasAdd|
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/Relu»
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool≤
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_8/Conv2D/ReadVariableOpў
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_8/Conv2D®
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp≠
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/Relu»
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPoolС
dropout_6/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_6/Identitys
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
flatten_2/ConstЫ
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_2/ReshapeІ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_6/MatMul/ReadVariableOp†
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/MatMul•
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_6/BiasAdd/ReadVariableOpҐ
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/ReluГ
dropout_7/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/IdentityІ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_7/MatMul/ReadVariableOp°
dense_7/MatMulMatMuldropout_7/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/MatMul•
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_7/BiasAdd/ReadVariableOpҐ
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/ReluГ
dropout_8/IdentityIdentitydense_7/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_8/Identity÷
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЌ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЌ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulЧ
IdentityIdentitydropout_8/Identity:output:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
”
c
*__inference_dropout_7_layer_call_fn_392671

inputs
identityИҐStatefulPartitionedCallб
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_3903732
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
б
E
)__inference_lambda_2_layer_call_fn_392373

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_3901062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѕ
°
)__inference_conv2d_8_layer_call_fn_392574

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_3901882
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ц
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_392579

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
з
F
*__inference_dropout_6_layer_call_fn_392596

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_3902002
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ћ
†
)__inference_conv2d_7_layer_call_fn_392554

inputs"
unknown: А
	unknown_0:	А
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_3901702
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Љњ
З
?__inference_CNN_layer_call_and_return_conditional_losses_391660
input_1H
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐ1sequential_2/batch_normalization_2/AssignNewValueҐ3sequential_2/batch_normalization_2/AssignNewValue_1ҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2м
#sequential_2/lambda_2/strided_sliceStridedSliceinput_12sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1–
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<25
3sequential_2/batch_normalization_2/FusedBatchNormV3с
1sequential_2/batch_normalization_2/AssignNewValueAssignVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource@sequential_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_2/batch_normalization_2/AssignNewValueэ
3sequential_2/batch_normalization_2/AssignNewValue_1AssignVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceDsequential_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0E^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_2/batch_normalization_2/AssignNewValue_1„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpЦ
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpа
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/BiasAddҐ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/Reluо
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpН
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpб
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/BiasAdd£
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/Reluп
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpН
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpб
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/BiasAdd£
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/Reluп
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolС
$sequential_2/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2&
$sequential_2/dropout_6/dropout/Constи
"sequential_2/dropout_6/dropout/MulMul-sequential_2/max_pooling2d_8/MaxPool:output:0-sequential_2/dropout_6/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2$
"sequential_2/dropout_6/dropout/Mul©
$sequential_2/dropout_6/dropout/ShapeShape-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_6/dropout/ShapeВ
;sequential_2/dropout_6/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_6/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02=
;sequential_2/dropout_6/dropout/random_uniform/RandomUniform£
-sequential_2/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-sequential_2/dropout_6/dropout/GreaterEqual/y£
+sequential_2/dropout_6/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_6/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_6/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2-
+sequential_2/dropout_6/dropout/GreaterEqualЌ
#sequential_2/dropout_6/dropout/CastCast/sequential_2/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2%
#sequential_2/dropout_6/dropout/Castя
$sequential_2/dropout_6/dropout/Mul_1Mul&sequential_2/dropout_6/dropout/Mul:z:0'sequential_2/dropout_6/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_2/dropout_6/dropout/Mul_1Н
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Constѕ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/dropout/Mul_1:z:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOp‘
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp÷
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/BiasAddШ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/ReluС
$sequential_2/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_2/dropout_7/dropout/ConstЏ
"sequential_2/dropout_7/dropout/MulMul'sequential_2/dense_6/Relu:activations:0-sequential_2/dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"sequential_2/dropout_7/dropout/Mul£
$sequential_2/dropout_7/dropout/ShapeShape'sequential_2/dense_6/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_7/dropout/Shapeъ
;sequential_2/dropout_7/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02=
;sequential_2/dropout_7/dropout/random_uniform/RandomUniform£
-sequential_2/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_2/dropout_7/dropout/GreaterEqual/yЫ
+sequential_2/dropout_7/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_7/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2-
+sequential_2/dropout_7/dropout/GreaterEqual≈
#sequential_2/dropout_7/dropout/CastCast/sequential_2/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2%
#sequential_2/dropout_7/dropout/Cast„
$sequential_2/dropout_7/dropout/Mul_1Mul&sequential_2/dropout_7/dropout/Mul:z:0'sequential_2/dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2&
$sequential_2/dropout_7/dropout/Mul_1ќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp’
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/dropout/Mul_1:z:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp÷
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/BiasAddШ
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/ReluС
$sequential_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_2/dropout_8/dropout/ConstЏ
"sequential_2/dropout_8/dropout/MulMul'sequential_2/dense_7/Relu:activations:0-sequential_2/dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"sequential_2/dropout_8/dropout/Mul£
$sequential_2/dropout_8/dropout/ShapeShape'sequential_2/dense_7/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_8/dropout/Shapeъ
;sequential_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02=
;sequential_2/dropout_8/dropout/random_uniform/RandomUniform£
-sequential_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_2/dropout_8/dropout/GreaterEqual/yЫ
+sequential_2/dropout_8/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_8/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2-
+sequential_2/dropout_8/dropout/GreaterEqual≈
#sequential_2/dropout_8/dropout/CastCast/sequential_2/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2%
#sequential_2/dropout_8/dropout/Cast„
$sequential_2/dropout_8/dropout/Mul_1Mul&sequential_2/dropout_8/dropout/Mul:z:0'sequential_2/dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2&
$sequential_2/dropout_8/dropout/Mul_1¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp≠
dense_8/MatMulMatMul(sequential_2/dropout_8/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp°
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/BiasAddy
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/Softmaxг
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЏ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЏ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulх
IdentityIdentitydense_8/Softmax:softmax:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp2^sequential_2/batch_normalization_2/AssignNewValue4^sequential_2/batch_normalization_2/AssignNewValue_1C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2f
1sequential_2/batch_normalization_2/AssignNewValue1sequential_2/batch_normalization_2/AssignNewValue2j
3sequential_2/batch_normalization_2/AssignNewValue_13sequential_2/batch_normalization_2/AssignNewValue_12И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
—
ь
-__inference_sequential_2_layer_call_fn_392299

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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИҐStatefulPartitionedCallЫ
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3906072
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠t
Џ
__inference_call_306754

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2л
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpЦ
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpа
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/BiasAddҐ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/Reluо
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpН
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpб
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/BiasAdd£
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/Reluп
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpН
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpб
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/BiasAdd£
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/Reluп
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolЄ
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_6/IdentityН
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Constѕ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOp‘
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp÷
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/BiasAddШ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/Relu™
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_7/Identityќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp’
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp÷
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/BiasAddШ
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/Relu™
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_8/Identity¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp≠
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp°
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/BiasAddy
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/Softmaxс
IdentityIdentitydense_8/Softmax:softmax:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЕУ
Э
?__inference_CNN_layer_call_and_return_conditional_losses_391549
input_1H
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2м
#sequential_2/lambda_2/strided_sliceStridedSliceinput_12sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpЦ
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpа
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/BiasAddҐ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/Reluо
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpН
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpб
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/BiasAdd£
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/Reluп
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpН
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpб
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/BiasAdd£
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/Reluп
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolЄ
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_6/IdentityН
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Constѕ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOp‘
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp÷
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/BiasAddШ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/Relu™
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_7/Identityќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp’
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp÷
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/BiasAddШ
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/Relu™
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_8/Identity¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp≠
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp°
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/BiasAddy
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/Softmaxг
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЏ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЏ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulЛ
IdentityIdentitydense_8/Softmax:softmax:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
у
c
*__inference_dropout_6_layer_call_fn_392601

inputs
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_3904122
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
«
F
*__inference_dropout_8_layer_call_fn_392725

inputs
identity…
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3902682
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
≠t
Џ
__inference_call_304624

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АG
3sequential_2_dense_6_matmul_readvariableop_resource:
ААC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_2/batch_normalization_2/ReadVariableOpҐ3sequential_2/batch_normalization_2/ReadVariableOp_1Ґ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_6/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_7/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_8/Conv2D/ReadVariableOpҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpѓ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_2/lambda_2/strided_slice/stack≥
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_2/lambda_2/strided_slice/stack_1≥
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_2/lambda_2/strided_slice/stack_2л
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2%
#sequential_2/lambda_2/strided_sliceЁ
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpг
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1Р
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3„
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_6/Conv2D/ReadVariableOpЦ
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
sequential_2/conv2d_6/Conv2Dќ
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpа
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/BiasAddҐ
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_2/conv2d_6/Reluо
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPoolЎ
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02-
+sequential_2/conv2d_7/Conv2D/ReadVariableOpН
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_7/Conv2Dѕ
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpб
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/BiasAdd£
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_7/Reluп
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPoolў
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOpН
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_2/conv2d_8/Conv2Dѕ
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpб
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/BiasAdd£
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_2/conv2d_8/Reluп
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolЄ
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_6/IdentityН
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
sequential_2/flatten_2/Constѕ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_2/flatten_2/Reshapeќ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOp‘
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/MatMulћ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp÷
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/BiasAddШ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_6/Relu™
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_7/Identityќ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp’
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/MatMulћ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp÷
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/BiasAddШ
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_2/dense_7/Relu™
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential_2/dropout_8/Identity¶
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_8/MatMul/ReadVariableOp≠
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/MatMul§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp°
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/BiasAddy
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_8/Softmaxс
IdentityIdentitydense_8/Softmax:softmax:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЧШ
÷
H__inference_sequential_2_layer_call_and_return_conditional_losses_392013

inputs;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: B
'conv2d_7_conv2d_readvariableop_resource: А7
(conv2d_7_biasadd_readvariableop_resource:	АC
'conv2d_8_conv2d_readvariableop_resource:АА7
(conv2d_8_biasadd_readvariableop_resource:	А:
&dense_6_matmul_readvariableop_resource:
АА6
'dense_6_biasadd_readvariableop_resource:	А:
&dense_7_matmul_readvariableop_resource:
АА6
'dense_7_biasadd_readvariableop_resource:	А
identityИҐ$batch_normalization_2/AssignNewValueҐ&batch_normalization_2/AssignNewValue_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐ0dense_6/kernel/Regularizer/Square/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ0dense_7/kernel/Regularizer/Square/ReadVariableOpХ
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_2/strided_slice/stackЩ
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_2/strided_slice/stack_1Щ
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_2/strided_slice/stack_2™
lambda_2/strided_sliceStridedSliceinputs%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask2
lambda_2/strided_sliceґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3lambda_2/strided_slice:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2(
&batch_normalization_2/FusedBatchNormV3∞
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueЉ
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOpв
conv2d_6/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_6/Relu«
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool±
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02 
conv2d_7/Conv2D/ReadVariableOpў
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_7/Conv2D®
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp≠
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/BiasAdd|
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_7/Relu»
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool≤
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_8/Conv2D/ReadVariableOpў
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_8/Conv2D®
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp≠
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_8/Relu»
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPoolw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_6/dropout/Constі
dropout_6/dropout/MulMul max_pooling2d_8/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/MulВ
dropout_6/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shapeџ
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_6/dropout/random_uniform/RandomUniformЙ
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_6/dropout/GreaterEqual/yп
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
dropout_6/dropout/GreaterEqual¶
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/CastЂ
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/Mul_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ 	  2
flatten_2/ConstЫ
flatten_2/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_2/ReshapeІ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_6/MatMul/ReadVariableOp†
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/MatMul•
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_6/BiasAdd/ReadVariableOpҐ
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_6/Reluw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/Const¶
dropout_7/dropout/MulMuldense_6/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Mul|
dropout_7/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape”
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_7/dropout/random_uniform/RandomUniformЙ
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/yз
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_7/dropout/GreaterEqualЮ
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Cast£
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Mul_1І
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_7/MatMul/ReadVariableOp°
dense_7/MatMulMatMuldropout_7/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/MatMul•
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_7/BiasAdd/ReadVariableOpҐ
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_7/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const¶
dropout_8/dropout/MulMuldense_7/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_8/dropout/Mul|
dropout_8/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape”
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЙ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/yз
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_8/dropout/GreaterEqualЮ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_8/dropout/Cast£
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_8/dropout/Mul_1÷
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЌ
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOpµ
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_6/kernel/Regularizer/SquareХ
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/ConstЇ
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/SumЙ
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_6/kernel/Regularizer/mul/xЉ
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mulЌ
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOpµ
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_7/kernel/Regularizer/SquareХ
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/ConstЇ
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/SumЙ
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_7/kernel/Regularizer/mul/xЉ
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mulз
IdentityIdentitydropout_8/dropout/Mul_1:z:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ы
≠
$__inference_CNN_layer_call_fn_391697
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
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

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
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_3908712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1"ћL
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
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:„В
ю


h2ptjl
_output
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+й&call_and_return_all_conditional_losses
к_default_save_signature
л__call__
	мcall"М	
_tf_keras_modelт{"name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 25, 25, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
™h
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
+н&call_and_return_all_conditional_losses
о__call__"√d
_tf_keras_sequential§d{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_2_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 25, 25, 2]}, "float32", "lambda_2_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_2_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
’

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
+п&call_and_return_all_conditional_losses
р__call__"Ѓ
_tf_keras_layerФ{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
л
!iter

"beta_1

#beta_2
	$decay
%learning_ratemЌmќ&mѕ'm–*m—+m“,m”-m‘.m’/m÷0m„1mЎ2mў3mЏvџv№&vЁ'vё*vя+vа,vб-vв.vг/vд0vе1vж2vз3vи"
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
 "
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
ќ
4non_trainable_variables
5layer_regularization_losses

6layers
	variables
regularization_losses
7layer_metrics
8metrics
trainable_variables
л__call__
к_default_save_signature
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
-
сserving_default"
signature_map
Ў
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+т&call_and_return_all_conditional_losses
у__call__"«
_tf_keras_layer≠{"name": "lambda_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
ƒ

=axis
	&gamma
'beta
(moving_mean
)moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"о
_tf_keras_layer‘{"name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 25, 25, 2]}}
†

*kernel
+bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"щ	
_tf_keras_layerя	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 25, 25, 2]}}
±
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"†
_tf_keras_layerЖ{"name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
‘


,kernel
-bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"≠	
_tf_keras_layerУ	{"name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 12, 12, 32]}}
±
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"†
_tf_keras_layerЖ{"name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
‘


.kernel
/bias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+ю&call_and_return_all_conditional_losses
€__call__"≠	
_tf_keras_layerУ	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 6, 6, 128]}}
±
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"†
_tf_keras_layerЖ{"name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
€
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"о
_tf_keras_layer‘{"name": "dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
Ш
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"З
_tf_keras_layerн{"name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
¶	

0kernel
1bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"€
_tf_keras_layerе{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 2304]}}
€
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"о
_tf_keras_layer‘{"name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
§	

2kernel
3bias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"э
_tf_keras_layerг{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
€
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"о
_tf_keras_layer‘{"name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
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
8
О0
П1
Р2"
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
∞
rnon_trainable_variables
slayer_regularization_losses

tlayers
	variables
regularization_losses
ulayer_metrics
vmetrics
trainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_8/kernel
:2dense_8/bias
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
∞
wnon_trainable_variables
xlayer_regularization_losses

ylayers
	variables
regularization_losses
zlayer_metrics
{metrics
trainable_variables
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
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
):' 2conv2d_6/kernel
: 2conv2d_6/bias
*:( А2conv2d_7/kernel
:А2conv2d_7/bias
+:)АА2conv2d_8/kernel
:А2conv2d_8/bias
": 
АА2dense_6/kernel
:А2dense_6/bias
": 
АА2dense_7/kernel
:А2dense_7/bias
.
(0
)1"
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
~non_trainable_variables
layer_regularization_losses
Аlayers
9	variables
:regularization_losses
Бlayer_metrics
Вmetrics
;trainable_variables
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
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
µ
Гnon_trainable_variables
 Дlayer_regularization_losses
Еlayers
>	variables
?regularization_losses
Жlayer_metrics
Зmetrics
@trainable_variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
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
µ
Иnon_trainable_variables
 Йlayer_regularization_losses
Кlayers
B	variables
Cregularization_losses
Лlayer_metrics
Мmetrics
Dtrainable_variables
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
Нnon_trainable_variables
 Оlayer_regularization_losses
Пlayers
F	variables
Gregularization_losses
Рlayer_metrics
Сmetrics
Htrainable_variables
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
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
µ
Тnon_trainable_variables
 Уlayer_regularization_losses
Фlayers
J	variables
Kregularization_losses
Хlayer_metrics
Цmetrics
Ltrainable_variables
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
Чnon_trainable_variables
 Шlayer_regularization_losses
Щlayers
N	variables
Oregularization_losses
Ъlayer_metrics
Ыmetrics
Ptrainable_variables
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
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
µ
Ьnon_trainable_variables
 Эlayer_regularization_losses
Юlayers
R	variables
Sregularization_losses
Яlayer_metrics
†metrics
Ttrainable_variables
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
°non_trainable_variables
 Ґlayer_regularization_losses
£layers
V	variables
Wregularization_losses
§layer_metrics
•metrics
Xtrainable_variables
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶non_trainable_variables
 Іlayer_regularization_losses
®layers
Z	variables
[regularization_losses
©layer_metrics
™metrics
\trainable_variables
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
Ђnon_trainable_variables
 ђlayer_regularization_losses
≠layers
^	variables
_regularization_losses
Ѓlayer_metrics
ѓmetrics
`trainable_variables
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
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
µ
∞non_trainable_variables
 ±layer_regularization_losses
≤layers
b	variables
cregularization_losses
≥layer_metrics
іmetrics
dtrainable_variables
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
µnon_trainable_variables
 ґlayer_regularization_losses
Јlayers
f	variables
gregularization_losses
Єlayer_metrics
єmetrics
htrainable_variables
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
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
µ
Їnon_trainable_variables
 їlayer_regularization_losses
Љlayers
j	variables
kregularization_losses
љlayer_metrics
Њmetrics
ltrainable_variables
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
њnon_trainable_variables
 јlayer_regularization_losses
Ѕlayers
n	variables
oregularization_losses
¬layer_metrics
√metrics
ptrainable_variables
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
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
 "
trackable_dict_wrapper
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
(
О0"
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
П0"
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
Р0"
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
&:$	А2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
.:,2"Adam/batch_normalization_2/gamma/m
-:+2!Adam/batch_normalization_2/beta/m
.:, 2Adam/conv2d_6/kernel/m
 : 2Adam/conv2d_6/bias/m
/:- А2Adam/conv2d_7/kernel/m
!:А2Adam/conv2d_7/bias/m
0:.АА2Adam/conv2d_8/kernel/m
!:А2Adam/conv2d_8/bias/m
':%
АА2Adam/dense_6/kernel/m
 :А2Adam/dense_6/bias/m
':%
АА2Adam/dense_7/kernel/m
 :А2Adam/dense_7/bias/m
&:$	А2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
.:,2"Adam/batch_normalization_2/gamma/v
-:+2!Adam/batch_normalization_2/beta/v
.:, 2Adam/conv2d_6/kernel/v
 : 2Adam/conv2d_6/bias/v
/:- А2Adam/conv2d_7/kernel/v
!:А2Adam/conv2d_7/bias/v
0:.АА2Adam/conv2d_8/kernel/v
!:А2Adam/conv2d_8/bias/v
':%
АА2Adam/dense_6/kernel/v
 :А2Adam/dense_6/bias/v
':%
АА2Adam/dense_7/kernel/v
 :А2Adam/dense_7/bias/v
Њ2ї
?__inference_CNN_layer_call_and_return_conditional_losses_391348
?__inference_CNN_layer_call_and_return_conditional_losses_391459
?__inference_CNN_layer_call_and_return_conditional_losses_391549
?__inference_CNN_layer_call_and_return_conditional_losses_391660і
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
з2д
!__inference__wrapped_model_389929Њ
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
input_1€€€€€€€€€
“2ѕ
$__inference_CNN_layer_call_fn_391697
$__inference_CNN_layer_call_fn_391734
$__inference_CNN_layer_call_fn_391771
$__inference_CNN_layer_call_fn_391808і
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
Д2Б
__inference_call_306610
__inference_call_306682
__inference_call_306754≥
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
о2л
H__inference_sequential_2_layer_call_and_return_conditional_losses_391909
H__inference_sequential_2_layer_call_and_return_conditional_losses_392013
H__inference_sequential_2_layer_call_and_return_conditional_losses_392096
H__inference_sequential_2_layer_call_and_return_conditional_losses_392200ј
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
В2€
-__inference_sequential_2_layer_call_fn_392233
-__inference_sequential_2_layer_call_fn_392266
-__inference_sequential_2_layer_call_fn_392299
-__inference_sequential_2_layer_call_fn_392332ј
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
н2к
C__inference_dense_8_layer_call_and_return_conditional_losses_392343Ґ
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
“2ѕ
(__inference_dense_8_layer_call_fn_392352Ґ
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
ЋB»
$__inference_signature_wrapper_391258input_1"Ф
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
“2ѕ
D__inference_lambda_2_layer_call_and_return_conditional_losses_392360
D__inference_lambda_2_layer_call_and_return_conditional_losses_392368ј
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
Ь2Щ
)__inference_lambda_2_layer_call_fn_392373
)__inference_lambda_2_layer_call_fn_392378ј
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
Ж2Г
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392396
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392414
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392432
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392450і
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
Ъ2Ч
6__inference_batch_normalization_2_layer_call_fn_392463
6__inference_batch_normalization_2_layer_call_fn_392476
6__inference_batch_normalization_2_layer_call_fn_392489
6__inference_batch_normalization_2_layer_call_fn_392502і
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
о2л
D__inference_conv2d_6_layer_call_and_return_conditional_losses_392525Ґ
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
”2–
)__inference_conv2d_6_layer_call_fn_392534Ґ
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
≥2∞
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_390061а
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
Ш2Х
0__inference_max_pooling2d_6_layer_call_fn_390067а
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
о2л
D__inference_conv2d_7_layer_call_and_return_conditional_losses_392545Ґ
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
”2–
)__inference_conv2d_7_layer_call_fn_392554Ґ
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
≥2∞
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_390073а
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
Ш2Х
0__inference_max_pooling2d_7_layer_call_fn_390079а
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
о2л
D__inference_conv2d_8_layer_call_and_return_conditional_losses_392565Ґ
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
”2–
)__inference_conv2d_8_layer_call_fn_392574Ґ
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
≥2∞
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_390085а
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
Ш2Х
0__inference_max_pooling2d_8_layer_call_fn_390091а
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
»2≈
E__inference_dropout_6_layer_call_and_return_conditional_losses_392579
E__inference_dropout_6_layer_call_and_return_conditional_losses_392591і
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
Т2П
*__inference_dropout_6_layer_call_fn_392596
*__inference_dropout_6_layer_call_fn_392601і
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
п2м
E__inference_flatten_2_layer_call_and_return_conditional_losses_392607Ґ
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
*__inference_flatten_2_layer_call_fn_392612Ґ
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
н2к
C__inference_dense_6_layer_call_and_return_conditional_losses_392635Ґ
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
“2ѕ
(__inference_dense_6_layer_call_fn_392644Ґ
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
»2≈
E__inference_dropout_7_layer_call_and_return_conditional_losses_392649
E__inference_dropout_7_layer_call_and_return_conditional_losses_392661і
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
Т2П
*__inference_dropout_7_layer_call_fn_392666
*__inference_dropout_7_layer_call_fn_392671і
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
н2к
C__inference_dense_7_layer_call_and_return_conditional_losses_392694Ґ
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
“2ѕ
(__inference_dense_7_layer_call_fn_392703Ґ
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
»2≈
E__inference_dropout_8_layer_call_and_return_conditional_losses_392708
E__inference_dropout_8_layer_call_and_return_conditional_losses_392720і
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
Т2П
*__inference_dropout_8_layer_call_fn_392725
*__inference_dropout_8_layer_call_fn_392730і
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
≥2∞
__inference_loss_fn_0_392741П
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
≥2∞
__inference_loss_fn_1_392752П
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
≥2∞
__inference_loss_fn_2_392763П
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
annotations™ *Ґ є
?__inference_CNN_layer_call_and_return_conditional_losses_391348v&'()*+,-./0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ є
?__inference_CNN_layer_call_and_return_conditional_losses_391459v&'()*+,-./0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
?__inference_CNN_layer_call_and_return_conditional_losses_391549w&'()*+,-./0123<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
?__inference_CNN_layer_call_and_return_conditional_losses_391660w&'()*+,-./0123<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Т
$__inference_CNN_layer_call_fn_391697j&'()*+,-./0123<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p 
™ "К€€€€€€€€€С
$__inference_CNN_layer_call_fn_391734i&'()*+,-./0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€С
$__inference_CNN_layer_call_fn_391771i&'()*+,-./0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ "К€€€€€€€€€Т
$__inference_CNN_layer_call_fn_391808j&'()*+,-./0123<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p
™ "К€€€€€€€€€І
!__inference__wrapped_model_389929Б&'()*+,-./01238Ґ5
.Ґ+
)К&
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392396Ц&'()MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392414Ц&'()MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ «
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392432r&'();Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ «
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_392450r&'();Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ƒ
6__inference_batch_normalization_2_layer_call_fn_392463Й&'()MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
6__inference_batch_normalization_2_layer_call_fn_392476Й&'()MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Я
6__inference_batch_normalization_2_layer_call_fn_392489e&'();Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ " К€€€€€€€€€Я
6__inference_batch_normalization_2_layer_call_fn_392502e&'();Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ " К€€€€€€€€€t
__inference_call_306610Y&'()*+,-./01233Ґ0
)Ґ&
 К
inputsА
p
™ "К	Аt
__inference_call_306682Y&'()*+,-./01233Ґ0
)Ґ&
 К
inputsА
p 
™ "К	АД
__inference_call_306754i&'()*+,-./0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€і
D__inference_conv2d_6_layer_call_and_return_conditional_losses_392525l*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ М
)__inference_conv2d_6_layer_call_fn_392534_*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€ µ
D__inference_conv2d_7_layer_call_and_return_conditional_losses_392545m,-7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Н
)__inference_conv2d_7_layer_call_fn_392554`,-7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "!К€€€€€€€€€Аґ
D__inference_conv2d_8_layer_call_and_return_conditional_losses_392565n./8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ О
)__inference_conv2d_8_layer_call_fn_392574a./8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€А•
C__inference_dense_6_layer_call_and_return_conditional_losses_392635^010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
(__inference_dense_6_layer_call_fn_392644Q010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А•
C__inference_dense_7_layer_call_and_return_conditional_losses_392694^230Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
(__inference_dense_7_layer_call_fn_392703Q230Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А§
C__inference_dense_8_layer_call_and_return_conditional_losses_392343]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
(__inference_dense_8_layer_call_fn_392352P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ј
E__inference_dropout_6_layer_call_and_return_conditional_losses_392579n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Ј
E__inference_dropout_6_layer_call_and_return_conditional_losses_392591n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ П
*__inference_dropout_6_layer_call_fn_392596a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€АП
*__inference_dropout_6_layer_call_fn_392601a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€АІ
E__inference_dropout_7_layer_call_and_return_conditional_losses_392649^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ І
E__inference_dropout_7_layer_call_and_return_conditional_losses_392661^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dropout_7_layer_call_fn_392666Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А
*__inference_dropout_7_layer_call_fn_392671Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АІ
E__inference_dropout_8_layer_call_and_return_conditional_losses_392708^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ І
E__inference_dropout_8_layer_call_and_return_conditional_losses_392720^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dropout_8_layer_call_fn_392725Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А
*__inference_dropout_8_layer_call_fn_392730Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АЂ
E__inference_flatten_2_layer_call_and_return_conditional_losses_392607b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Г
*__inference_flatten_2_layer_call_fn_392612U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€АЄ
D__inference_lambda_2_layer_call_and_return_conditional_losses_392360p?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€

 
p 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Є
D__inference_lambda_2_layer_call_and_return_conditional_losses_392368p?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€

 
p
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Р
)__inference_lambda_2_layer_call_fn_392373c?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€

 
p 
™ " К€€€€€€€€€Р
)__inference_lambda_2_layer_call_fn_392378c?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€

 
p
™ " К€€€€€€€€€;
__inference_loss_fn_0_392741*Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_3927520Ґ

Ґ 
™ "К ;
__inference_loss_fn_2_3927632Ґ

Ґ 
™ "К о
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_390061ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_6_layer_call_fn_390067СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_390073ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_7_layer_call_fn_390079СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_390085ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_8_layer_call_fn_390091СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€≈
H__inference_sequential_2_layer_call_and_return_conditional_losses_391909y&'()*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ≈
H__inference_sequential_2_layer_call_and_return_conditional_losses_392013y&'()*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ќ
H__inference_sequential_2_layer_call_and_return_conditional_losses_392096Б&'()*+,-./0123GҐD
=Ґ:
0К-
lambda_2_input€€€€€€€€€
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ќ
H__inference_sequential_2_layer_call_and_return_conditional_losses_392200Б&'()*+,-./0123GҐD
=Ґ:
0К-
lambda_2_input€€€€€€€€€
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ •
-__inference_sequential_2_layer_call_fn_392233t&'()*+,-./0123GҐD
=Ґ:
0К-
lambda_2_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€АЭ
-__inference_sequential_2_layer_call_fn_392266l&'()*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€АЭ
-__inference_sequential_2_layer_call_fn_392299l&'()*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€А•
-__inference_sequential_2_layer_call_fn_392332t&'()*+,-./0123GҐD
=Ґ:
0К-
lambda_2_input€€€€€€€€€
p

 
™ "К€€€€€€€€€Аµ
$__inference_signature_wrapper_391258М&'()*+,-./0123CҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€