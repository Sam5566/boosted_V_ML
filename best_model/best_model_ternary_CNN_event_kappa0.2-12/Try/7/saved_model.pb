и╞
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718чж
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	А*
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
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
Д
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
: *
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
: *
dtype0
Е
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_22/kernel
~
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*'
_output_shapes
: А*
dtype0
u
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_22/bias
n
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_23/kernel

$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_23/bias
n
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes	
:А*
dtype0
|
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_21/kernel
u
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:А*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:А*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
Ы
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
Й
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_23/kernel/m
В
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	А*
dtype0
А
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
Ь
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m
Х
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m
У
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_21/kernel/m
Л
+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_21/bias/m
{
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes
: *
dtype0
У
Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_22/kernel/m
М
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_22/bias/m
|
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_23/kernel/m
Н
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_23/bias/m
|
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_21/kernel/m
Г
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_22/kernel/m
Г
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_23/kernel/v
В
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	А*
dtype0
А
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
Ь
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v
Х
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v
У
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_21/kernel/v
Л
+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_21/bias/v
{
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes
: *
dtype0
У
Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_22/kernel/v
М
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_22/bias/v
|
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_23/kernel/v
Н
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_23/bias/v
|
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_21/kernel/v
Г
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_22/kernel/v
Г
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
╥`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н`
valueГ`BА` B∙_
К

h2ptjl
_output
	optimizer
regularization_losses
trainable_variables
	variables
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
╪
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m╧'m╨(m╤)m╥*m╙+m╘,m╒-m╓.m╫/m╪0m┘1m┌v█v▄&v▌'v▐(v▀)vр*vс+vт,vу-vф.vх/vц0vч1vш
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
н
4non_trainable_variables
regularization_losses
trainable_variables
	variables

5layers
6layer_regularization_losses
7layer_metrics
8metrics
 
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
Ч
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
н
rnon_trainable_variables
regularization_losses
trainable_variables
	variables

slayers
tlayer_regularization_losses
ulayer_metrics
vmetrics
NL
VARIABLE_VALUEdense_23/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_23/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
wnon_trainable_variables
regularization_losses
trainable_variables
	variables

xlayers
ylayer_regularization_losses
zlayer_metrics
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
VARIABLE_VALUEbatch_normalization_7/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_7/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_21/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_21/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_22/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_22/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_23/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_23/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_21/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_21/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_22/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_22/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_7/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_7/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE

20
31

0
1
 
 

|0
}1
 
 
 
░
~non_trainable_variables
9regularization_losses
:trainable_variables
;	variables

layers
 Аlayer_regularization_losses
Бlayer_metrics
Вmetrics
 
 

&0
'1

&0
'1
22
33
▓
Гnon_trainable_variables
>regularization_losses
?trainable_variables
@	variables
Дlayers
 Еlayer_regularization_losses
Жlayer_metrics
Зmetrics
 

(0
)1

(0
)1
▓
Иnon_trainable_variables
Bregularization_losses
Ctrainable_variables
D	variables
Йlayers
 Кlayer_regularization_losses
Лlayer_metrics
Мmetrics
 
 
 
▓
Нnon_trainable_variables
Fregularization_losses
Gtrainable_variables
H	variables
Оlayers
 Пlayer_regularization_losses
Рlayer_metrics
Сmetrics
 

*0
+1

*0
+1
▓
Тnon_trainable_variables
Jregularization_losses
Ktrainable_variables
L	variables
Уlayers
 Фlayer_regularization_losses
Хlayer_metrics
Цmetrics
 
 
 
▓
Чnon_trainable_variables
Nregularization_losses
Otrainable_variables
P	variables
Шlayers
 Щlayer_regularization_losses
Ъlayer_metrics
Ыmetrics
 

,0
-1

,0
-1
▓
Ьnon_trainable_variables
Rregularization_losses
Strainable_variables
T	variables
Эlayers
 Юlayer_regularization_losses
Яlayer_metrics
аmetrics
 
 
 
▓
бnon_trainable_variables
Vregularization_losses
Wtrainable_variables
X	variables
вlayers
 гlayer_regularization_losses
дlayer_metrics
еmetrics
 
 
 
▓
жnon_trainable_variables
Zregularization_losses
[trainable_variables
\	variables
зlayers
 иlayer_regularization_losses
йlayer_metrics
кmetrics
 
 
 
▓
лnon_trainable_variables
^regularization_losses
_trainable_variables
`	variables
мlayers
 нlayer_regularization_losses
оlayer_metrics
пmetrics
 

.0
/1

.0
/1
▓
░non_trainable_variables
bregularization_losses
ctrainable_variables
d	variables
▒layers
 ▓layer_regularization_losses
│layer_metrics
┤metrics
 
 
 
▓
╡non_trainable_variables
fregularization_losses
gtrainable_variables
h	variables
╢layers
 ╖layer_regularization_losses
╕layer_metrics
╣metrics
 

00
11

00
11
▓
║non_trainable_variables
jregularization_losses
ktrainable_variables
l	variables
╗layers
 ╝layer_regularization_losses
╜layer_metrics
╛metrics
 
 
 
▓
┐non_trainable_variables
nregularization_losses
otrainable_variables
p	variables
└layers
 ┴layer_regularization_losses
┬layer_metrics
├metrics

20
31
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
VARIABLE_VALUEAdam/dense_23/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_21/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_21/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_22/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_22/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_23/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_23/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_21/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_21/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_22/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_22/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_23/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_21/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_21/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_22/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_22/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_23/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_23/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_21/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_21/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_22/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_22/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
б
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*
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
GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1281353
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOpConst*B
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
 __inference__traced_save_1283040
┴
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_7/gammabatch_normalization_7/betaconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/bias!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancetotalcounttotal_1count_1Adam/dense_23/kernel/mAdam/dense_23/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/vAdam/dense_23/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/v*A
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
#__inference__traced_restore_1283209Ж╡
└
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282509

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
╚
╠
"__inference__wrapped_model_1280024
input_1
cnn_1279990:
cnn_1279992:
cnn_1279994:
cnn_1279996:%
cnn_1279998: 
cnn_1280000: &
cnn_1280002: А
cnn_1280004:	А'
cnn_1280006:АА
cnn_1280008:	А
cnn_1280010:
АА
cnn_1280012:	А
cnn_1280014:
АА
cnn_1280016:	А
cnn_1280018:	А
cnn_1280020:
identityИвCNN/StatefulPartitionedCallо
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_1279990cnn_1279992cnn_1279994cnn_1279996cnn_1279998cnn_1280000cnn_1280002cnn_1280004cnn_1280006cnn_1280008cnn_1280010cnn_1280012cnn_1280014cnn_1280016cnn_1280018cnn_1280020*
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
GPU2 *0J 8В *!
fR
__inference_call_11873232
CNN/StatefulPartitionedCallЦ
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^CNN/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2:
CNN/StatefulPartitionedCallCNN/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
∙Ф
╕
@__inference_CNN_layer_call_and_return_conditional_losses_1281443

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpф
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
sequential_7/conv2d_21/BiasAddе
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_7/conv2d_21/Reluё
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpх
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_22/BiasAddж
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_22/ReluЄ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpх
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_23/BiasAddж
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_23/ReluЄ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_21/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╫
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/BiasAddЫ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/Reluн
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_22/Identity╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/BiasAddЫ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/Reluн
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_23/Identityй
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmaxц
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul▌
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul▌
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulЫ
IdentityIdentitydense_23/Softmax:softmax:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
№Ф
╣
@__inference_CNN_layer_call_and_return_conditional_losses_1281644
input_1H
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2ь
#sequential_7/lambda_7/strided_sliceStridedSliceinput_12sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpф
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
sequential_7/conv2d_21/BiasAddе
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_7/conv2d_21/Reluё
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpх
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_22/BiasAddж
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_22/ReluЄ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpх
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_23/BiasAddж
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_23/ReluЄ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_21/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╫
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/BiasAddЫ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/Reluн
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_22/Identity╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/BiasAddЫ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/Reluн
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_23/Identityй
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmaxц
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul▌
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul▌
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulЫ
IdentityIdentitydense_23/Softmax:softmax:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
°
e
G__inference_dropout_23_layer_call_and_return_conditional_losses_1282803

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
ї┴
г
@__inference_CNN_layer_call_and_return_conditional_losses_1281755
input_1H
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpв1sequential_7/batch_normalization_7/AssignNewValueв3sequential_7/batch_normalization_7/AssignNewValue_1вBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2ь
#sequential_7/lambda_7/strided_sliceStridedSliceinput_12sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_7/batch_normalization_7/FusedBatchNormV3ё
1sequential_7/batch_normalization_7/AssignNewValueAssignVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource@sequential_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_7/batch_normalization_7/AssignNewValue¤
3sequential_7/batch_normalization_7/AssignNewValue_1AssignVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceDsequential_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0E^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_7/batch_normalization_7/AssignNewValue_1┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpф
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
sequential_7/conv2d_21/BiasAddе
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_7/conv2d_21/Reluё
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpх
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_22/BiasAddж
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_22/ReluЄ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpх
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_23/BiasAddж
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_23/ReluЄ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPoolУ
%sequential_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2'
%sequential_7/dropout_21/dropout/Constь
#sequential_7/dropout_21/dropout/MulMul.sequential_7/max_pooling2d_23/MaxPool:output:0.sequential_7/dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         А2%
#sequential_7/dropout_21/dropout/Mulм
%sequential_7/dropout_21/dropout/ShapeShape.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_21/dropout/ShapeЕ
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_7/dropout_21/dropout/GreaterEqual/yз
,sequential_7/dropout_21/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_21/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2.
,sequential_7/dropout_21/dropout/GreaterEqual╨
$sequential_7/dropout_21/dropout/CastCast0sequential_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2&
$sequential_7/dropout_21/dropout/Castу
%sequential_7/dropout_21/dropout/Mul_1Mul'sequential_7/dropout_21/dropout/Mul:z:0(sequential_7/dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2'
%sequential_7/dropout_21/dropout/Mul_1Н
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/dropout/Mul_1:z:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╫
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/BiasAddЫ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/ReluУ
%sequential_7/dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_22/dropout/Const▐
#sequential_7/dropout_22/dropout/MulMul(sequential_7/dense_21/Relu:activations:0.sequential_7/dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_7/dropout_22/dropout/Mulж
%sequential_7/dropout_22/dropout/ShapeShape(sequential_7/dense_21/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_22/dropout/Shape¤
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_22/dropout/GreaterEqual/yЯ
,sequential_7/dropout_22/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_22/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_7/dropout_22/dropout/GreaterEqual╚
$sequential_7/dropout_22/dropout/CastCast0sequential_7/dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_7/dropout_22/dropout/Cast█
%sequential_7/dropout_22/dropout/Mul_1Mul'sequential_7/dropout_22/dropout/Mul:z:0(sequential_7/dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_7/dropout_22/dropout/Mul_1╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/dropout/Mul_1:z:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/BiasAddЫ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/ReluУ
%sequential_7/dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_23/dropout/Const▐
#sequential_7/dropout_23/dropout/MulMul(sequential_7/dense_22/Relu:activations:0.sequential_7/dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_7/dropout_23/dropout/Mulж
%sequential_7/dropout_23/dropout/ShapeShape(sequential_7/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_23/dropout/Shape¤
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_23/dropout/GreaterEqual/yЯ
,sequential_7/dropout_23/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_23/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_7/dropout_23/dropout/GreaterEqual╚
$sequential_7/dropout_23/dropout/CastCast0sequential_7/dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_7/dropout_23/dropout/Cast█
%sequential_7/dropout_23/dropout/Mul_1Mul'sequential_7/dropout_23/dropout/Mul:z:0(sequential_7/dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_7/dropout_23/dropout/Mul_1й
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmaxц
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul▌
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul▌
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulЕ	
IdentityIdentitydense_23/Softmax:softmax:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp2^sequential_7/batch_normalization_7/AssignNewValue4^sequential_7/batch_normalization_7/AssignNewValue_1C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2f
1sequential_7/batch_normalization_7/AssignNewValue1sequential_7/batch_normalization_7/AssignNewValue2j
3sequential_7/batch_normalization_7/AssignNewValue_13sequential_7/batch_normalization_7/AssignNewValue_12И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
Єs
є
__inference_call_1189309

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2у
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:А*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1║
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:А:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpС
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp▄
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:А 2 
sequential_7/conv2d_21/BiasAddЭ
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*'
_output_shapes
:А 2
sequential_7/conv2d_21/Reluщ
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*'
_output_shapes
:А *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_7/conv2d_22/BiasAddЮ
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_7/conv2d_22/Reluъ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_7/conv2d_23/BiasAddЮ
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_7/conv2d_23/Reluъ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPool│
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_7/dropout_21/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╚
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0* 
_output_shapes
:
АА2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╧
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp╥
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_21/BiasAddУ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_21/Reluе
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_7/dropout_22/Identity╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp╤
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp╥
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_22/BiasAddУ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_22/Reluе
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_7/dropout_23/Identityй
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOpй
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЭ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_23/BiasAddt
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_23/SoftmaxЎ
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:А: : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
ж
╥
7__inference_batch_normalization_7_layer_call_fn_1282597

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
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12805732
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╢
f
G__inference_dropout_23_layer_call_and_return_conditional_losses_1282815

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
и
╥
7__inference_batch_normalization_7_layer_call_fn_1282584

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
:         *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12802202
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╫
e
,__inference_dropout_22_layer_call_fn_1282766

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
G__inference_dropout_22_layer_call_and_return_conditional_losses_12804682
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
Ы
╝
__inference_loss_fn_0_1282836U
;conv2d_21_kernel_regularizer_square_readvariableop_resource: 
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpь
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_21_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mulЬ
IdentityIdentity$conv2d_21/kernel/Regularizer/mul:z:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp
└
┤
F__inference_conv2d_21_layer_call_and_return_conditional_losses_1280247

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
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
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Relu╧
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╫
e
,__inference_dropout_23_layer_call_fn_1282825

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
G__inference_dropout_23_layer_call_and_return_conditional_losses_12804352
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
М
Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1280046

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
М
Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282491

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
Эv
ж
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282191
lambda_7_input;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: C
(conv2d_22_conv2d_readvariableop_resource: А8
)conv2d_22_biasadd_readvariableop_resource:	АD
(conv2d_23_conv2d_readvariableop_resource:АА8
)conv2d_23_biasadd_readvariableop_resource:	А;
'dense_21_matmul_readvariableop_resource:
АА7
(dense_21_biasadd_readvariableop_resource:	А;
'dense_22_matmul_readvariableop_resource:
АА7
(dense_22_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_21/BiasAdd/ReadVariableOpвconv2d_21/Conv2D/ReadVariableOpв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв conv2d_22/BiasAdd/ReadVariableOpвconv2d_22/Conv2D/ReadVariableOpв conv2d_23/BiasAdd/ReadVariableOpвconv2d_23/Conv2D/ReadVariableOpвdense_21/BiasAdd/ReadVariableOpвdense_21/MatMul/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpвdense_22/BiasAdd/ReadVariableOpвdense_22/MatMul/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
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
:         *

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3│
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpх
conv2d_21/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_21/Conv2Dк
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp░
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_21/Relu╩
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool┤
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_22/Conv2D/ReadVariableOp▌
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_22/Conv2Dл
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool╡
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_23/Conv2D/ReadVariableOp▌
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_23/Conv2Dл
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolФ
dropout_21/IdentityIdentity!max_pooling2d_23/MaxPool:output:0*
T0*0
_output_shapes
:         А2
dropout_21/Identitys
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_21/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_7/Reshapeк
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_21/MatMul/ReadVariableOpг
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/MatMulи
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_21/BiasAdd/ReadVariableOpж
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_21/ReluЖ
dropout_22/IdentityIdentitydense_21/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_22/Identityк
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_22/MatMul/ReadVariableOpе
dense_22/MatMulMatMuldropout_22/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/MatMulи
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_22/BiasAdd/ReadVariableOpж
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_22/ReluЖ
dropout_23/IdentityIdentitydense_22/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_23/Identity┘
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╨
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╨
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulе
IdentityIdentitydropout_23/Identity:output:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         
(
_user_specified_namelambda_7_input
°
e
G__inference_dropout_22_layer_call_and_return_conditional_losses_1280333

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
║
н
E__inference_dense_22_layer_call_and_return_conditional_losses_1280352

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpП
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
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ў
f
G__inference_dropout_21_layer_call_and_return_conditional_losses_1282686

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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╢
f
G__inference_dropout_23_layer_call_and_return_conditional_losses_1280435

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
°
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282545

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
7:         :::::*
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
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Уa
З	
I__inference_sequential_7_layer_call_and_return_conditional_losses_1280702

inputs+
batch_normalization_7_1280642:+
batch_normalization_7_1280644:+
batch_normalization_7_1280646:+
batch_normalization_7_1280648:+
conv2d_21_1280651: 
conv2d_21_1280653: ,
conv2d_22_1280657: А 
conv2d_22_1280659:	А-
conv2d_23_1280663:АА 
conv2d_23_1280665:	А$
dense_21_1280671:
АА
dense_21_1280673:	А$
dense_22_1280677:
АА
dense_22_1280679:	А
identityИв-batch_normalization_7/StatefulPartitionedCallв!conv2d_21/StatefulPartitionedCallв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв!conv2d_22/StatefulPartitionedCallв!conv2d_23/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв1dense_21/kernel/Regularizer/Square/ReadVariableOpв dense_22/StatefulPartitionedCallв1dense_22/kernel/Regularizer/Square/ReadVariableOpв"dropout_21/StatefulPartitionedCallв"dropout_22/StatefulPartitionedCallв"dropout_23/StatefulPartitionedCallт
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_12806002
lambda_7/PartitionedCall└
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0batch_normalization_7_1280642batch_normalization_7_1280644batch_normalization_7_1280646batch_normalization_7_1280648*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12805732/
-batch_normalization_7/StatefulPartitionedCall┘
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_21_1280651conv2d_21_1280653*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_12802472#
!conv2d_21/StatefulPartitionedCallЮ
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_12801562"
 max_pooling2d_21/PartitionedCall═
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_22_1280657conv2d_22_1280659*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_12802652#
!conv2d_22/StatefulPartitionedCallЯ
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_12801682"
 max_pooling2d_22/PartitionedCall═
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_1280663conv2d_23_1280665*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_23_layer_call_and_return_conditional_losses_12802832#
!conv2d_23/StatefulPartitionedCallЯ
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_12801802"
 max_pooling2d_23/PartitionedCallд
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_12805072$
"dropout_21/StatefulPartitionedCallГ
flatten_7/PartitionedCallPartitionedCall+dropout_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_12803032
flatten_7/PartitionedCall╣
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_1280671dense_21_1280673*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_12803222"
 dense_21/StatefulPartitionedCall┴
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
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
G__inference_dropout_22_layer_call_and_return_conditional_losses_12804682$
"dropout_22/StatefulPartitionedCall┬
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_22_1280677dense_22_1280679*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_12803522"
 dense_22/StatefulPartitionedCall┴
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
G__inference_dropout_23_layer_call_and_return_conditional_losses_12804352$
"dropout_23/StatefulPartitionedCall┬
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_1280651*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╣
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_1280671* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╣
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_1280677* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulю
IdentityIdentity+dropout_23/StatefulPartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_1282455

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
:         *

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_21_layer_call_and_return_conditional_losses_1280295

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
└
┤
F__inference_conv2d_21_layer_call_and_return_conditional_losses_1282620

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
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
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Relu╧
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Є┴
в
@__inference_CNN_layer_call_and_return_conditional_losses_1281554

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpв1sequential_7/batch_normalization_7/AssignNewValueв3sequential_7/batch_normalization_7/AssignNewValue_1вBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_7/batch_normalization_7/FusedBatchNormV3ё
1sequential_7/batch_normalization_7/AssignNewValueAssignVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource@sequential_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_7/batch_normalization_7/AssignNewValue¤
3sequential_7/batch_normalization_7/AssignNewValue_1AssignVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceDsequential_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0E^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_7/batch_normalization_7/AssignNewValue_1┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpф
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
sequential_7/conv2d_21/BiasAddе
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_7/conv2d_21/Reluё
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpх
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_22/BiasAddж
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_22/ReluЄ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpх
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_23/BiasAddж
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_23/ReluЄ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPoolУ
%sequential_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2'
%sequential_7/dropout_21/dropout/Constь
#sequential_7/dropout_21/dropout/MulMul.sequential_7/max_pooling2d_23/MaxPool:output:0.sequential_7/dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         А2%
#sequential_7/dropout_21/dropout/Mulм
%sequential_7/dropout_21/dropout/ShapeShape.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_21/dropout/ShapeЕ
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_7/dropout_21/dropout/GreaterEqual/yз
,sequential_7/dropout_21/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_21/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2.
,sequential_7/dropout_21/dropout/GreaterEqual╨
$sequential_7/dropout_21/dropout/CastCast0sequential_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2&
$sequential_7/dropout_21/dropout/Castу
%sequential_7/dropout_21/dropout/Mul_1Mul'sequential_7/dropout_21/dropout/Mul:z:0(sequential_7/dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2'
%sequential_7/dropout_21/dropout/Mul_1Н
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/dropout/Mul_1:z:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╫
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/BiasAddЫ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/ReluУ
%sequential_7/dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_22/dropout/Const▐
#sequential_7/dropout_22/dropout/MulMul(sequential_7/dense_21/Relu:activations:0.sequential_7/dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_7/dropout_22/dropout/Mulж
%sequential_7/dropout_22/dropout/ShapeShape(sequential_7/dense_21/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_22/dropout/Shape¤
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_22/dropout/GreaterEqual/yЯ
,sequential_7/dropout_22/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_22/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_7/dropout_22/dropout/GreaterEqual╚
$sequential_7/dropout_22/dropout/CastCast0sequential_7/dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_7/dropout_22/dropout/Cast█
%sequential_7/dropout_22/dropout/Mul_1Mul'sequential_7/dropout_22/dropout/Mul:z:0(sequential_7/dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_7/dropout_22/dropout/Mul_1╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/dropout/Mul_1:z:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/BiasAddЫ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/ReluУ
%sequential_7/dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_23/dropout/Const▐
#sequential_7/dropout_23/dropout/MulMul(sequential_7/dense_22/Relu:activations:0.sequential_7/dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_7/dropout_23/dropout/Mulж
%sequential_7/dropout_23/dropout/ShapeShape(sequential_7/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_23/dropout/Shape¤
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_23/dropout/GreaterEqual/yЯ
,sequential_7/dropout_23/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_23/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_7/dropout_23/dropout/GreaterEqual╚
$sequential_7/dropout_23/dropout/CastCast0sequential_7/dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_7/dropout_23/dropout/Cast█
%sequential_7/dropout_23/dropout/Mul_1Mul'sequential_7/dropout_23/dropout/Mul:z:0(sequential_7/dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_7/dropout_23/dropout/Mul_1й
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmaxц
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul▌
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul▌
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulЕ	
IdentityIdentitydense_23/Softmax:softmax:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp2^sequential_7/batch_normalization_7/AssignNewValue4^sequential_7/batch_normalization_7/AssignNewValue_1C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2f
1sequential_7/batch_normalization_7/AssignNewValue1sequential_7/batch_normalization_7/AssignNewValue2j
3sequential_7/batch_normalization_7/AssignNewValue_13sequential_7/batch_normalization_7/AssignNewValue_12И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╣

ў
E__inference_dense_23_layer_call_and_return_conditional_losses_1282438

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
Ц2
║
@__inference_CNN_layer_call_and_return_conditional_losses_1281106

inputs"
sequential_7_1281053:"
sequential_7_1281055:"
sequential_7_1281057:"
sequential_7_1281059:.
sequential_7_1281061: "
sequential_7_1281063: /
sequential_7_1281065: А#
sequential_7_1281067:	А0
sequential_7_1281069:АА#
sequential_7_1281071:	А(
sequential_7_1281073:
АА#
sequential_7_1281075:	А(
sequential_7_1281077:
АА#
sequential_7_1281079:	А#
dense_23_1281082:	А
dense_23_1281084:
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpв dense_23/StatefulPartitionedCallв$sequential_7/StatefulPartitionedCall╧
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_1281053sequential_7_1281055sequential_7_1281057sequential_7_1281059sequential_7_1281061sequential_7_1281063sequential_7_1281065sequential_7_1281067sequential_7_1281069sequential_7_1281071sequential_7_1281073sequential_7_1281075sequential_7_1281077sequential_7_1281079*
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
GPU2 *0J 8В *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_12807022&
$sequential_7/StatefulPartitionedCall├
 dense_23/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0dense_23_1281082dense_23_1281084*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_12809412"
 dense_23/StatefulPartitionedCall┼
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_1281061*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╜
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_1281073* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╜
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_1281077* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulф
IdentityIdentity)dense_23/StatefulPartitionedCall:output:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_1282463

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
:         *

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
щ
┤
__inference_loss_fn_2_1282858N
:dense_22_kernel_regularizer_square_readvariableop_resource:
АА
identityИв1dense_22/kernel/Regularizer/Square/ReadVariableOpу
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_22_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulЪ
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
═Ъ
Ў
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282295
lambda_7_input;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: C
(conv2d_22_conv2d_readvariableop_resource: А8
)conv2d_22_biasadd_readvariableop_resource:	АD
(conv2d_23_conv2d_readvariableop_resource:АА8
)conv2d_23_biasadd_readvariableop_resource:	А;
'dense_21_matmul_readvariableop_resource:
АА7
(dense_21_biasadd_readvariableop_resource:	А;
'dense_22_matmul_readvariableop_resource:
АА7
(dense_22_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_21/BiasAdd/ReadVariableOpвconv2d_21/Conv2D/ReadVariableOpв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв conv2d_22/BiasAdd/ReadVariableOpвconv2d_22/Conv2D/ReadVariableOpв conv2d_23/BiasAdd/ReadVariableOpвconv2d_23/Conv2D/ReadVariableOpвdense_21/BiasAdd/ReadVariableOpвdense_21/MatMul/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpвdense_22/BiasAdd/ReadVariableOpвdense_22/MatMul/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
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
:         *

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpх
conv2d_21/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_21/Conv2Dк
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp░
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_21/Relu╩
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool┤
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_22/Conv2D/ReadVariableOp▌
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_22/Conv2Dл
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool╡
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_23/Conv2D/ReadVariableOp▌
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_23/Conv2Dл
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPooly
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_21/dropout/Const╕
dropout_21/dropout/MulMul!max_pooling2d_23/MaxPool:output:0!dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_21/dropout/MulЕ
dropout_21/dropout/ShapeShape!max_pooling2d_23/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shape▐
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformЛ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_21/dropout/GreaterEqual/yє
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2!
dropout_21/dropout/GreaterEqualй
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_21/dropout/Castп
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_21/dropout/Mul_1s
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_21/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_7/Reshapeк
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_21/MatMul/ReadVariableOpг
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/MatMulи
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_21/BiasAdd/ReadVariableOpж
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_21/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_22/dropout/Constк
dropout_22/dropout/MulMuldense_21/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape╓
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_22/dropout/random_uniform/RandomUniformЛ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_22/dropout/GreaterEqual/yы
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_22/dropout/GreaterEqualб
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_22/dropout/Castз
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_22/dropout/Mul_1к
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_22/MatMul/ReadVariableOpе
dense_22/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/MatMulи
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_22/BiasAdd/ReadVariableOpж
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_22/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_23/dropout/Constк
dropout_23/dropout/MulMuldense_22/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shape╓
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_23/dropout/random_uniform/RandomUniformЛ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_23/dropout/GreaterEqual/yы
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_23/dropout/GreaterEqualб
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_23/dropout/Castз
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_23/dropout/Mul_1┘
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╨
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╨
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulї
IdentityIdentitydropout_23/dropout/Mul_1:z:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         
(
_user_specified_namelambda_7_input
╠
а
+__inference_conv2d_21_layer_call_fn_1282629

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
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_12802472
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_1280600

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
:         *

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╙
¤
.__inference_sequential_7_layer_call_fn_1282394

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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЬ
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
GPU2 *0J 8В *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_12807022
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
у
F
*__inference_lambda_7_layer_call_fn_1282468

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
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_12802012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_21_layer_call_fn_1280162

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
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_12801562
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
╦
H
,__inference_dropout_22_layer_call_fn_1282761

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
G__inference_dropout_22_layer_call_and_return_conditional_losses_12803332
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
╨
в
+__inference_conv2d_22_layer_call_fn_1282649

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
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_12802652
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
е
Ш
*__inference_dense_23_layer_call_fn_1282447

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
E__inference_dense_23_layer_call_and_return_conditional_losses_12809412
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
Єs
є
__inference_call_1189381

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2у
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:А*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1║
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:А:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpС
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp▄
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:А 2 
sequential_7/conv2d_21/BiasAddЭ
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*'
_output_shapes
:А 2
sequential_7/conv2d_21/Reluщ
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*'
_output_shapes
:А *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_7/conv2d_22/BiasAddЮ
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_7/conv2d_22/Reluъ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_7/conv2d_23/BiasAddЮ
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_7/conv2d_23/Reluъ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPool│
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_7/dropout_21/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╚
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0* 
_output_shapes
:
АА2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╧
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp╥
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_21/BiasAddУ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_21/Reluе
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_7/dropout_22/Identity╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp╤
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp╥
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_22/BiasAddУ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_22/Reluе
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_7/dropout_23/Identityй
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOpй
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЭ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_23/BiasAddt
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_23/SoftmaxЎ
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:А: : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
║
н
E__inference_dense_21_layer_call_and_return_conditional_losses_1282730

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╦
H
,__inference_dropout_23_layer_call_fn_1282820

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
G__inference_dropout_23_layer_call_and_return_conditional_losses_12803632
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
п\
Ш
I__inference_sequential_7_layer_call_and_return_conditional_losses_1280384

inputs+
batch_normalization_7_1280221:+
batch_normalization_7_1280223:+
batch_normalization_7_1280225:+
batch_normalization_7_1280227:+
conv2d_21_1280248: 
conv2d_21_1280250: ,
conv2d_22_1280266: А 
conv2d_22_1280268:	А-
conv2d_23_1280284:АА 
conv2d_23_1280286:	А$
dense_21_1280323:
АА
dense_21_1280325:	А$
dense_22_1280353:
АА
dense_22_1280355:	А
identityИв-batch_normalization_7/StatefulPartitionedCallв!conv2d_21/StatefulPartitionedCallв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв!conv2d_22/StatefulPartitionedCallв!conv2d_23/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв1dense_21/kernel/Regularizer/Square/ReadVariableOpв dense_22/StatefulPartitionedCallв1dense_22/kernel/Regularizer/Square/ReadVariableOpт
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_12802012
lambda_7/PartitionedCall┬
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0batch_normalization_7_1280221batch_normalization_7_1280223batch_normalization_7_1280225batch_normalization_7_1280227*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12802202/
-batch_normalization_7/StatefulPartitionedCall┘
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_21_1280248conv2d_21_1280250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_12802472#
!conv2d_21/StatefulPartitionedCallЮ
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_12801562"
 max_pooling2d_21/PartitionedCall═
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_22_1280266conv2d_22_1280268*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_12802652#
!conv2d_22/StatefulPartitionedCallЯ
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_12801682"
 max_pooling2d_22/PartitionedCall═
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_1280284conv2d_23_1280286*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_23_layer_call_and_return_conditional_losses_12802832#
!conv2d_23/StatefulPartitionedCallЯ
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_12801802"
 max_pooling2d_23/PartitionedCallМ
dropout_21/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_12802952
dropout_21/PartitionedCall√
flatten_7/PartitionedCallPartitionedCall#dropout_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_12803032
flatten_7/PartitionedCall╣
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_1280323dense_21_1280325*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_12803222"
 dense_21/StatefulPartitionedCallД
dropout_22/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
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
G__inference_dropout_22_layer_call_and_return_conditional_losses_12803332
dropout_22/PartitionedCall║
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_22_1280353dense_22_1280355*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_12803522"
 dense_22/StatefulPartitionedCallД
dropout_23/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
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
G__inference_dropout_23_layer_call_and_return_conditional_losses_12803632
dropout_23/PartitionedCall┬
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_1280248*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╣
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_1280323* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╣
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_1280353* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulў
IdentityIdentity#dropout_23/PartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─
Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1280220

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
7:         :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
°
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1280573

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
7:         :::::*
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
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
в
В
F__inference_conv2d_23_layer_call_and_return_conditional_losses_1282660

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
:         А*
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
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Еv
Ю
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282004

inputs;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: C
(conv2d_22_conv2d_readvariableop_resource: А8
)conv2d_22_biasadd_readvariableop_resource:	АD
(conv2d_23_conv2d_readvariableop_resource:АА8
)conv2d_23_biasadd_readvariableop_resource:	А;
'dense_21_matmul_readvariableop_resource:
АА7
(dense_21_biasadd_readvariableop_resource:	А;
'dense_22_matmul_readvariableop_resource:
АА7
(dense_22_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_21/BiasAdd/ReadVariableOpвconv2d_21/Conv2D/ReadVariableOpв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв conv2d_22/BiasAdd/ReadVariableOpвconv2d_22/Conv2D/ReadVariableOpв conv2d_23/BiasAdd/ReadVariableOpвconv2d_23/Conv2D/ReadVariableOpвdense_21/BiasAdd/ReadVariableOpвdense_21/MatMul/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpвdense_22/BiasAdd/ReadVariableOpвdense_22/MatMul/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2к
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3│
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpх
conv2d_21/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_21/Conv2Dк
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp░
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_21/Relu╩
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool┤
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_22/Conv2D/ReadVariableOp▌
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_22/Conv2Dл
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool╡
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_23/Conv2D/ReadVariableOp▌
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_23/Conv2Dл
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolФ
dropout_21/IdentityIdentity!max_pooling2d_23/MaxPool:output:0*
T0*0
_output_shapes
:         А2
dropout_21/Identitys
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_21/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_7/Reshapeк
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_21/MatMul/ReadVariableOpг
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/MatMulи
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_21/BiasAdd/ReadVariableOpж
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_21/ReluЖ
dropout_22/IdentityIdentitydense_21/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_22/Identityк
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_22/MatMul/ReadVariableOpе
dense_22/MatMulMatMuldropout_22/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/MatMulи
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_22/BiasAdd/ReadVariableOpж
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_22/ReluЖ
dropout_23/IdentityIdentitydense_22/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_23/Identity┘
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╨
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╨
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulе
IdentityIdentitydropout_23/Identity:output:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ё
╥
7__inference_batch_normalization_7_layer_call_fn_1282558

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12800462
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
╡Ъ
ю
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282108

inputs;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: C
(conv2d_22_conv2d_readvariableop_resource: А8
)conv2d_22_biasadd_readvariableop_resource:	АD
(conv2d_23_conv2d_readvariableop_resource:АА8
)conv2d_23_biasadd_readvariableop_resource:	А;
'dense_21_matmul_readvariableop_resource:
АА7
(dense_21_biasadd_readvariableop_resource:	А;
'dense_22_matmul_readvariableop_resource:
АА7
(dense_22_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_21/BiasAdd/ReadVariableOpвconv2d_21/Conv2D/ReadVariableOpв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв conv2d_22/BiasAdd/ReadVariableOpвconv2d_22/Conv2D/ReadVariableOpв conv2d_23/BiasAdd/ReadVariableOpвconv2d_23/Conv2D/ReadVariableOpвdense_21/BiasAdd/ReadVariableOpвdense_21/MatMul/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpвdense_22/BiasAdd/ReadVariableOpвdense_22/MatMul/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2к
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpх
conv2d_21/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_21/Conv2Dк
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp░
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_21/Relu╩
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool┤
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_22/Conv2D/ReadVariableOp▌
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_22/Conv2Dл
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool╡
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_23/Conv2D/ReadVariableOp▌
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_23/Conv2Dл
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPooly
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_21/dropout/Const╕
dropout_21/dropout/MulMul!max_pooling2d_23/MaxPool:output:0!dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_21/dropout/MulЕ
dropout_21/dropout/ShapeShape!max_pooling2d_23/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shape▐
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformЛ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_21/dropout/GreaterEqual/yє
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2!
dropout_21/dropout/GreaterEqualй
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_21/dropout/Castп
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_21/dropout/Mul_1s
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_21/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_7/Reshapeк
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_21/MatMul/ReadVariableOpг
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/MatMulи
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_21/BiasAdd/ReadVariableOpж
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_21/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_22/dropout/Constк
dropout_22/dropout/MulMuldense_21/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape╓
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_22/dropout/random_uniform/RandomUniformЛ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_22/dropout/GreaterEqual/yы
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_22/dropout/GreaterEqualб
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_22/dropout/Castз
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_22/dropout/Mul_1к
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_22/MatMul/ReadVariableOpе
dense_22/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/MatMulи
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_22/BiasAdd/ReadVariableOpж
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_22/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_23/dropout/Constк
dropout_23/dropout/MulMuldense_22/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shape╓
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_23/dropout/random_uniform/RandomUniformЛ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_23/dropout/GreaterEqual/yы
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_23/dropout/GreaterEqualб
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_23/dropout/Castз
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_23/dropout/Mul_1┘
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╨
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╨
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulї
IdentityIdentitydropout_23/dropout/Mul_1:z:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
н
i
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1280168

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
╒
¤
.__inference_sequential_7_layer_call_fn_1282361

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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЮ
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
GPU2 *0J 8В *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_12803842
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
└
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1280090

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
 
о
%__inference_signature_wrapper_1281353
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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallУ
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
GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_12800242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
Э
о
%__inference_CNN_layer_call_fn_1281792
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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCall▒
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
GPU2 *0J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_12809662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
Ъ
н
%__inference_CNN_layer_call_fn_1281829

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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCall░
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
GPU2 *0J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_12809662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╢
f
G__inference_dropout_22_layer_call_and_return_conditional_losses_1280468

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
Ы
о
%__inference_CNN_layer_call_fn_1281903
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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallп
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
GPU2 *0J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_12811062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
ў
e
,__inference_dropout_21_layer_call_fn_1282696

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_12805072
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
й
Ъ
*__inference_dense_21_layer_call_fn_1282739

inputs
unknown:
АА
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
E__inference_dense_21_layer_call_and_return_conditional_losses_12803222
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_22_layer_call_fn_1280174

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
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_12801682
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
н
i
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1280180

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
й
Ъ
*__inference_dense_22_layer_call_fn_1282798

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
E__inference_dense_22_layer_call_and_return_conditional_losses_12803522
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
дт
╩!
#__inference__traced_restore_1283209
file_prefix3
 assignvariableop_dense_23_kernel:	А.
 assignvariableop_1_dense_23_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_7_gamma:;
-assignvariableop_8_batch_normalization_7_beta:=
#assignvariableop_9_conv2d_21_kernel: 0
"assignvariableop_10_conv2d_21_bias: ?
$assignvariableop_11_conv2d_22_kernel: А1
"assignvariableop_12_conv2d_22_bias:	А@
$assignvariableop_13_conv2d_23_kernel:АА1
"assignvariableop_14_conv2d_23_bias:	А7
#assignvariableop_15_dense_21_kernel:
АА0
!assignvariableop_16_dense_21_bias:	А7
#assignvariableop_17_dense_22_kernel:
АА0
!assignvariableop_18_dense_22_bias:	АC
5assignvariableop_19_batch_normalization_7_moving_mean:G
9assignvariableop_20_batch_normalization_7_moving_variance:#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_23_kernel_m:	А6
(assignvariableop_26_adam_dense_23_bias_m:D
6assignvariableop_27_adam_batch_normalization_7_gamma_m:C
5assignvariableop_28_adam_batch_normalization_7_beta_m:E
+assignvariableop_29_adam_conv2d_21_kernel_m: 7
)assignvariableop_30_adam_conv2d_21_bias_m: F
+assignvariableop_31_adam_conv2d_22_kernel_m: А8
)assignvariableop_32_adam_conv2d_22_bias_m:	АG
+assignvariableop_33_adam_conv2d_23_kernel_m:АА8
)assignvariableop_34_adam_conv2d_23_bias_m:	А>
*assignvariableop_35_adam_dense_21_kernel_m:
АА7
(assignvariableop_36_adam_dense_21_bias_m:	А>
*assignvariableop_37_adam_dense_22_kernel_m:
АА7
(assignvariableop_38_adam_dense_22_bias_m:	А=
*assignvariableop_39_adam_dense_23_kernel_v:	А6
(assignvariableop_40_adam_dense_23_bias_v:D
6assignvariableop_41_adam_batch_normalization_7_gamma_v:C
5assignvariableop_42_adam_batch_normalization_7_beta_v:E
+assignvariableop_43_adam_conv2d_21_kernel_v: 7
)assignvariableop_44_adam_conv2d_21_bias_v: F
+assignvariableop_45_adam_conv2d_22_kernel_v: А8
)assignvariableop_46_adam_conv2d_22_bias_v:	АG
+assignvariableop_47_adam_conv2d_23_kernel_v:АА8
)assignvariableop_48_adam_conv2d_23_bias_v:	А>
*assignvariableop_49_adam_dense_21_kernel_v:
АА7
(assignvariableop_50_adam_dense_21_bias_v:	А>
*assignvariableop_51_adam_dense_22_kernel_v:
АА7
(assignvariableop_52_adam_dense_22_bias_v:	А
identity_54ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╬
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*┌
value╨B═6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*
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

Identity_9и
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_21_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10к
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_21_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11м
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_22_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_22_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13м
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_23_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_23_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15л
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_21_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16й
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_21_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17л
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_22_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_22_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╜
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_7_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┴
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_7_moving_varianceIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_23_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_23_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_7_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╜
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_7_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_21_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_21_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_22_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_22_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_23_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_23_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_21_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36░
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_21_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_22_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_22_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_23_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_23_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╛
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_7_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╜
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_7_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_21_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_21_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_22_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_22_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_23_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_23_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▓
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_21_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50░
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_21_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▓
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_22_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52░
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_22_bias_vIdentity_52:output:0"/device:CPU:0*
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
эj
╚
 __inference__traced_save_1283040
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
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop.
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
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop
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
ShardedFilename╚
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*┌
value╨B═6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЎ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*ч
_input_shapes╒
╥: :	А:: : : : : ::: : : А:А:АА:А:
АА:А:
АА:А::: : : : :	А:::: : : А:А:АА:А:
АА:А:
АА:А:	А:::: : : А:А:АА:А:
АА:А:
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
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!
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
:А:&$"
 
_output_shapes
:
АА:!%
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
АА:!3
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
Ш
н
%__inference_CNN_layer_call_fn_1281866

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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallо
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
GPU2 *0J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_12811062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
у
F
*__inference_lambda_7_layer_call_fn_1282473

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
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_12806002
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_21_layer_call_and_return_conditional_losses_1282674

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_23_layer_call_fn_1280186

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
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_12801802
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
╣

ў
E__inference_dense_23_layer_call_and_return_conditional_losses_1280941

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
э
Е
.__inference_sequential_7_layer_call_fn_1282328
lambda_7_input
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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCalllambda_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2 *0J 8В *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_12803842
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         
(
_user_specified_namelambda_7_input
║
н
E__inference_dense_21_layer_call_and_return_conditional_losses_1280322

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ъ
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_1280303

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╢
f
G__inference_dropout_22_layer_call_and_return_conditional_losses_1282756

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
тu
є
__inference_call_1189453

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpф
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
sequential_7/conv2d_21/BiasAddе
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_7/conv2d_21/Reluё
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpх
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_22/BiasAddж
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_22/ReluЄ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpх
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_23/BiasAddж
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_23/ReluЄ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_21/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╫
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/BiasAddЫ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/Reluн
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_22/Identity╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/BiasAddЫ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/Reluн
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_23/Identityй
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmax■
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
║
н
E__inference_dense_22_layer_call_and_return_conditional_losses_1282789

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpП
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
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
щ
┤
__inference_loss_fn_1_1282847N
:dense_21_kernel_regularizer_square_readvariableop_resource:
АА
identityИв1dense_21/kernel/Regularizer/Square/ReadVariableOpу
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_21_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulЪ
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
в
В
F__inference_conv2d_23_layer_call_and_return_conditional_losses_1280283

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
:         А*
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
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
─
Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282527

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
7:         :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╙
г
+__inference_conv2d_23_layer_call_fn_1282669

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
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_23_layer_call_and_return_conditional_losses_12802832
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ю
Б
F__inference_conv2d_22_layer_call_and_return_conditional_losses_1282640

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
:         А*
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
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ю
Б
F__inference_conv2d_22_layer_call_and_return_conditional_losses_1280265

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
:         А*
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
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ў
f
G__inference_dropout_21_layer_call_and_return_conditional_losses_1280507

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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ы
H
,__inference_dropout_21_layer_call_fn_1282691

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_12802952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┘
G
+__inference_flatten_7_layer_call_fn_1282707

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_12803032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
н
i
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1280156

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
G__inference_dropout_22_layer_call_and_return_conditional_losses_1282744

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
7__inference_batch_normalization_7_layer_call_fn_1282571

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12800902
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
°
e
G__inference_dropout_23_layer_call_and_return_conditional_losses_1280363

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
ы
Е
.__inference_sequential_7_layer_call_fn_1282427
lambda_7_input
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
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCalllambda_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2 *0J 8В *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_12807022
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         
(
_user_specified_namelambda_7_input
тu
є
__inference_call_1187323

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_21_conv2d_readvariableop_resource: D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: P
5sequential_7_conv2d_22_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_22_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_23_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_23_biasadd_readvariableop_resource:	АH
4sequential_7_dense_21_matmul_readvariableop_resource:
ААD
5sequential_7_dense_21_biasadd_readvariableop_resource:	АH
4sequential_7_dense_22_matmul_readvariableop_resource:
ААD
5sequential_7_dense_22_biasadd_readvariableop_resource:	А:
'dense_23_matmul_readvariableop_resource:	А6
(dense_23_biasadd_readvariableop_resource:
identityИвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_21/BiasAdd/ReadVariableOpв,sequential_7/conv2d_21/Conv2D/ReadVariableOpв-sequential_7/conv2d_22/BiasAdd/ReadVariableOpв,sequential_7/conv2d_22/Conv2D/ReadVariableOpв-sequential_7/conv2d_23/BiasAdd/ReadVariableOpв,sequential_7/conv2d_23/Conv2D/ReadVariableOpв,sequential_7/dense_21/BiasAdd/ReadVariableOpв+sequential_7/dense_21/MatMul/ReadVariableOpв,sequential_7/dense_22/BiasAdd/ReadVariableOpв+sequential_7/dense_22/MatMul/ReadVariableOpп
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
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_21/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_7/conv2d_21/Conv2D╤
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpф
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
sequential_7/conv2d_21/BiasAddе
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_7/conv2d_21/Reluё
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPool█
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_22/Conv2D╥
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpх
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_22/BiasAddж
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_22/ReluЄ
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPool▄
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_23/Conv2D╥
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpх
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_23/BiasAddж
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_23/ReluЄ
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_23/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_21/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp╫
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/MatMul╧
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/BiasAddЫ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_21/Reluн
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_22/Identity╤
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/MatMul╧
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/BiasAddЫ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_22/Reluн
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_23/Identityй
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmax■
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_1280201

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
:         *

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_1282702

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ш2
║
@__inference_CNN_layer_call_and_return_conditional_losses_1280966

inputs"
sequential_7_1280901:"
sequential_7_1280903:"
sequential_7_1280905:"
sequential_7_1280907:.
sequential_7_1280909: "
sequential_7_1280911: /
sequential_7_1280913: А#
sequential_7_1280915:	А0
sequential_7_1280917:АА#
sequential_7_1280919:	А(
sequential_7_1280921:
АА#
sequential_7_1280923:	А(
sequential_7_1280925:
АА#
sequential_7_1280927:	А#
dense_23_1280942:	А
dense_23_1280944:
identityИв2conv2d_21/kernel/Regularizer/Square/ReadVariableOpв1dense_21/kernel/Regularizer/Square/ReadVariableOpв1dense_22/kernel/Regularizer/Square/ReadVariableOpв dense_23/StatefulPartitionedCallв$sequential_7/StatefulPartitionedCall╤
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_1280901sequential_7_1280903sequential_7_1280905sequential_7_1280907sequential_7_1280909sequential_7_1280911sequential_7_1280913sequential_7_1280915sequential_7_1280917sequential_7_1280919sequential_7_1280921sequential_7_1280923sequential_7_1280925sequential_7_1280927*
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
GPU2 *0J 8В *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_12803842&
$sequential_7/StatefulPartitionedCall├
 dense_23/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0dense_23_1280942dense_23_1280944*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_12809412"
 dense_23/StatefulPartitionedCall┼
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_1280909*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/Squareб
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/Const┬
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/SumН
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_21/kernel/Regularizer/mul/x─
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul╜
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_1280921* 
_output_shapes
:
АА*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOp╕
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_21/kernel/Regularizer/SquareЧ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/Const╛
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumЛ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╜
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_1280925* 
_output_shapes
:
АА*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOp╕
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_22/kernel/Regularizer/SquareЧ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/Const╛
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumЛ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulф
IdentityIdentity)dense_23/StatefulPartitionedCall:output:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
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
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:эД
■


h2ptjl
_output
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+щ&call_and_return_all_conditional_losses
ъ_default_save_signature
ы__call__
	ьcall"М	
_tf_keras_modelЄ{"name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 15, 15, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
└h
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
+э&call_and_return_all_conditional_losses
ю__call__"┘d
_tf_keras_sequential║d{"name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 15, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_7_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 15, 15, 2]}, "float32", "lambda_7_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 15, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_7_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
╫

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"░
_tf_keras_layerЦ{"name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
ы
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m╧'m╨(m╤)m╥*m╙+m╘,m╒-m╓.m╫/m╪0m┘1m┌v█v▄&v▌'v▐(v▀)vр*vс+vт,vу-vф.vх/vц0vч1vш"
	optimizer
 "
trackable_list_wrapper
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
╬
4non_trainable_variables
regularization_losses
trainable_variables
	variables

5layers
6layer_regularization_losses
7layer_metrics
8metrics
ы__call__
ъ_default_save_signature
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
-
ёserving_default"
signature_map
╪
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+Є&call_and_return_all_conditional_losses
є__call__"╟
_tf_keras_layerн{"name": "lambda_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
─

=axis
	&gamma
'beta
2moving_mean
3moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+Ї&call_and_return_all_conditional_losses
ї__call__"ю
_tf_keras_layer╘{"name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 15, 15, 2]}}
в

(kernel
)bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"√	
_tf_keras_layerс	{"name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 15, 15, 2]}}
│
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+°&call_and_return_all_conditional_losses
∙__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
╘


*kernel
+bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+·&call_and_return_all_conditional_losses
√__call__"н	
_tf_keras_layerУ	{"name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 7, 7, 32]}}
│
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+№&call_and_return_all_conditional_losses
¤__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
╓


,kernel
-bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+■&call_and_return_all_conditional_losses
 __call__"п	
_tf_keras_layerХ	{"name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 3, 3, 128]}}
│
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
Б
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"Ё
_tf_keras_layer╓{"name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
Ш
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"З
_tf_keras_layerэ{"name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
ж	

.kernel
/bias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__" 
_tf_keras_layerх{"name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 256]}}
Б
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"Ё
_tf_keras_layer╓{"name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
ж	

0kernel
1bias
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
+К&call_and_return_all_conditional_losses
Л__call__" 
_tf_keras_layerх{"name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Б
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"Ё
_tf_keras_layer╓{"name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
8
О0
П1
Р2"
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
░
rnon_trainable_variables
regularization_losses
trainable_variables
	variables

slayers
tlayer_regularization_losses
ulayer_metrics
vmetrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_23/kernel
:2dense_23/bias
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
░
wnon_trainable_variables
regularization_losses
trainable_variables
	variables

xlayers
ylayer_regularization_losses
zlayer_metrics
{metrics
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
*:( 2conv2d_21/kernel
: 2conv2d_21/bias
+:) А2conv2d_22/kernel
:А2conv2d_22/bias
,:*АА2conv2d_23/kernel
:А2conv2d_23/bias
#:!
АА2dense_21/kernel
:А2dense_21/bias
#:!
АА2dense_22/kernel
:А2dense_22/bias
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
.
20
31"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
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
│
~non_trainable_variables
9regularization_losses
:trainable_variables
;	variables

layers
 Аlayer_regularization_losses
Бlayer_metrics
Вmetrics
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
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
╡
Гnon_trainable_variables
>regularization_losses
?trainable_variables
@	variables
Дlayers
 Еlayer_regularization_losses
Жlayer_metrics
Зmetrics
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
(
О0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
╡
Иnon_trainable_variables
Bregularization_losses
Ctrainable_variables
D	variables
Йlayers
 Кlayer_regularization_losses
Лlayer_metrics
Мmetrics
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Нnon_trainable_variables
Fregularization_losses
Gtrainable_variables
H	variables
Оlayers
 Пlayer_regularization_losses
Рlayer_metrics
Сmetrics
∙__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
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
╡
Тnon_trainable_variables
Jregularization_losses
Ktrainable_variables
L	variables
Уlayers
 Фlayer_regularization_losses
Хlayer_metrics
Цmetrics
√__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Чnon_trainable_variables
Nregularization_losses
Otrainable_variables
P	variables
Шlayers
 Щlayer_regularization_losses
Ъlayer_metrics
Ыmetrics
¤__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
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
╡
Ьnon_trainable_variables
Rregularization_losses
Strainable_variables
T	variables
Эlayers
 Юlayer_regularization_losses
Яlayer_metrics
аmetrics
 __call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
бnon_trainable_variables
Vregularization_losses
Wtrainable_variables
X	variables
вlayers
 гlayer_regularization_losses
дlayer_metrics
еmetrics
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
╡
жnon_trainable_variables
Zregularization_losses
[trainable_variables
\	variables
зlayers
 иlayer_regularization_losses
йlayer_metrics
кmetrics
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
╡
лnon_trainable_variables
^regularization_losses
_trainable_variables
`	variables
мlayers
 нlayer_regularization_losses
оlayer_metrics
пmetrics
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
(
П0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
╡
░non_trainable_variables
bregularization_losses
ctrainable_variables
d	variables
▒layers
 ▓layer_regularization_losses
│layer_metrics
┤metrics
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
╡
╡non_trainable_variables
fregularization_losses
gtrainable_variables
h	variables
╢layers
 ╖layer_regularization_losses
╕layer_metrics
╣metrics
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
(
Р0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
╡
║non_trainable_variables
jregularization_losses
ktrainable_variables
l	variables
╗layers
 ╝layer_regularization_losses
╜layer_metrics
╛metrics
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
╡
┐non_trainable_variables
nregularization_losses
otrainable_variables
p	variables
└layers
 ┴layer_regularization_losses
┬layer_metrics
├metrics
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
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
20
31"
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
 "
trackable_dict_wrapper
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
':%	А2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
.:,2"Adam/batch_normalization_7/gamma/m
-:+2!Adam/batch_normalization_7/beta/m
/:- 2Adam/conv2d_21/kernel/m
!: 2Adam/conv2d_21/bias/m
0:. А2Adam/conv2d_22/kernel/m
": А2Adam/conv2d_22/bias/m
1:/АА2Adam/conv2d_23/kernel/m
": А2Adam/conv2d_23/bias/m
(:&
АА2Adam/dense_21/kernel/m
!:А2Adam/dense_21/bias/m
(:&
АА2Adam/dense_22/kernel/m
!:А2Adam/dense_22/bias/m
':%	А2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v
.:,2"Adam/batch_normalization_7/gamma/v
-:+2!Adam/batch_normalization_7/beta/v
/:- 2Adam/conv2d_21/kernel/v
!: 2Adam/conv2d_21/bias/v
0:. А2Adam/conv2d_22/kernel/v
": А2Adam/conv2d_22/bias/v
1:/АА2Adam/conv2d_23/kernel/v
": А2Adam/conv2d_23/bias/v
(:&
АА2Adam/dense_21/kernel/v
!:А2Adam/dense_21/bias/v
(:&
АА2Adam/dense_22/kernel/v
!:А2Adam/dense_22/bias/v
┬2┐
@__inference_CNN_layer_call_and_return_conditional_losses_1281443
@__inference_CNN_layer_call_and_return_conditional_losses_1281554
@__inference_CNN_layer_call_and_return_conditional_losses_1281644
@__inference_CNN_layer_call_and_return_conditional_losses_1281755┤
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
"__inference__wrapped_model_1280024╛
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
input_1         
╓2╙
%__inference_CNN_layer_call_fn_1281792
%__inference_CNN_layer_call_fn_1281829
%__inference_CNN_layer_call_fn_1281866
%__inference_CNN_layer_call_fn_1281903┤
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
З2Д
__inference_call_1189309
__inference_call_1189381
__inference_call_1189453│
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
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282004
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282108
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282191
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282295└
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
.__inference_sequential_7_layer_call_fn_1282328
.__inference_sequential_7_layer_call_fn_1282361
.__inference_sequential_7_layer_call_fn_1282394
.__inference_sequential_7_layer_call_fn_1282427└
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
E__inference_dense_23_layer_call_and_return_conditional_losses_1282438в
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
*__inference_dense_23_layer_call_fn_1282447в
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
%__inference_signature_wrapper_1281353input_1"Ф
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
E__inference_lambda_7_layer_call_and_return_conditional_losses_1282455
E__inference_lambda_7_layer_call_and_return_conditional_losses_1282463└
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
*__inference_lambda_7_layer_call_fn_1282468
*__inference_lambda_7_layer_call_fn_1282473└
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282491
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282509
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282527
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282545┤
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
7__inference_batch_normalization_7_layer_call_fn_1282558
7__inference_batch_normalization_7_layer_call_fn_1282571
7__inference_batch_normalization_7_layer_call_fn_1282584
7__inference_batch_normalization_7_layer_call_fn_1282597┤
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
F__inference_conv2d_21_layer_call_and_return_conditional_losses_1282620в
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
+__inference_conv2d_21_layer_call_fn_1282629в
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
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1280156р
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
2__inference_max_pooling2d_21_layer_call_fn_1280162р
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
F__inference_conv2d_22_layer_call_and_return_conditional_losses_1282640в
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
+__inference_conv2d_22_layer_call_fn_1282649в
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
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1280168р
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
2__inference_max_pooling2d_22_layer_call_fn_1280174р
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
F__inference_conv2d_23_layer_call_and_return_conditional_losses_1282660в
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
+__inference_conv2d_23_layer_call_fn_1282669в
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
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1280180р
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
2__inference_max_pooling2d_23_layer_call_fn_1280186р
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
G__inference_dropout_21_layer_call_and_return_conditional_losses_1282674
G__inference_dropout_21_layer_call_and_return_conditional_losses_1282686┤
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
,__inference_dropout_21_layer_call_fn_1282691
,__inference_dropout_21_layer_call_fn_1282696┤
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
F__inference_flatten_7_layer_call_and_return_conditional_losses_1282702в
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
+__inference_flatten_7_layer_call_fn_1282707в
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
E__inference_dense_21_layer_call_and_return_conditional_losses_1282730в
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
*__inference_dense_21_layer_call_fn_1282739в
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
G__inference_dropout_22_layer_call_and_return_conditional_losses_1282744
G__inference_dropout_22_layer_call_and_return_conditional_losses_1282756┤
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
,__inference_dropout_22_layer_call_fn_1282761
,__inference_dropout_22_layer_call_fn_1282766┤
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
E__inference_dense_22_layer_call_and_return_conditional_losses_1282789в
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
*__inference_dense_22_layer_call_fn_1282798в
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
G__inference_dropout_23_layer_call_and_return_conditional_losses_1282803
G__inference_dropout_23_layer_call_and_return_conditional_losses_1282815┤
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
,__inference_dropout_23_layer_call_fn_1282820
,__inference_dropout_23_layer_call_fn_1282825┤
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
__inference_loss_fn_0_1282836П
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
__inference_loss_fn_1_1282847П
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
__inference_loss_fn_2_1282858П
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
annotationsк *в ║
@__inference_CNN_layer_call_and_return_conditional_losses_1281443v&'23()*+,-./01;в8
1в.
(К%
inputs         
p 
к "%в"
К
0         
Ъ ║
@__inference_CNN_layer_call_and_return_conditional_losses_1281554v&'23()*+,-./01;в8
1в.
(К%
inputs         
p
к "%в"
К
0         
Ъ ╗
@__inference_CNN_layer_call_and_return_conditional_losses_1281644w&'23()*+,-./01<в9
2в/
)К&
input_1         
p 
к "%в"
К
0         
Ъ ╗
@__inference_CNN_layer_call_and_return_conditional_losses_1281755w&'23()*+,-./01<в9
2в/
)К&
input_1         
p
к "%в"
К
0         
Ъ У
%__inference_CNN_layer_call_fn_1281792j&'23()*+,-./01<в9
2в/
)К&
input_1         
p 
к "К         Т
%__inference_CNN_layer_call_fn_1281829i&'23()*+,-./01;в8
1в.
(К%
inputs         
p 
к "К         Т
%__inference_CNN_layer_call_fn_1281866i&'23()*+,-./01;в8
1в.
(К%
inputs         
p
к "К         У
%__inference_CNN_layer_call_fn_1281903j&'23()*+,-./01<в9
2в/
)К&
input_1         
p
к "К         и
"__inference__wrapped_model_1280024Б&'23()*+,-./018в5
.в+
)К&
input_1         
к "3к0
.
output_1"К
output_1         э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282491Ц&'23MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282509Ц&'23MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╚
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282527r&'23;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ ╚
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1282545r&'23;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ ┼
7__inference_batch_normalization_7_layer_call_fn_1282558Й&'23MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ┼
7__inference_batch_normalization_7_layer_call_fn_1282571Й&'23MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           а
7__inference_batch_normalization_7_layer_call_fn_1282584e&'23;в8
1в.
(К%
inputs         
p 
к " К         а
7__inference_batch_normalization_7_layer_call_fn_1282597e&'23;в8
1в.
(К%
inputs         
p
к " К         u
__inference_call_1189309Y&'23()*+,-./013в0
)в&
 К
inputsА
p
к "К	Аu
__inference_call_1189381Y&'23()*+,-./013в0
)в&
 К
inputsА
p 
к "К	АЕ
__inference_call_1189453i&'23()*+,-./01;в8
1в.
(К%
inputs         
p 
к "К         ╢
F__inference_conv2d_21_layer_call_and_return_conditional_losses_1282620l()7в4
-в*
(К%
inputs         
к "-в*
#К 
0          
Ъ О
+__inference_conv2d_21_layer_call_fn_1282629_()7в4
-в*
(К%
inputs         
к " К          ╖
F__inference_conv2d_22_layer_call_and_return_conditional_losses_1282640m*+7в4
-в*
(К%
inputs          
к ".в+
$К!
0         А
Ъ П
+__inference_conv2d_22_layer_call_fn_1282649`*+7в4
-в*
(К%
inputs          
к "!К         А╕
F__inference_conv2d_23_layer_call_and_return_conditional_losses_1282660n,-8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Р
+__inference_conv2d_23_layer_call_fn_1282669a,-8в5
.в+
)К&
inputs         А
к "!К         Аз
E__inference_dense_21_layer_call_and_return_conditional_losses_1282730^./0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ 
*__inference_dense_21_layer_call_fn_1282739Q./0в-
&в#
!К
inputs         А
к "К         Аз
E__inference_dense_22_layer_call_and_return_conditional_losses_1282789^010в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ 
*__inference_dense_22_layer_call_fn_1282798Q010в-
&в#
!К
inputs         А
к "К         Аж
E__inference_dense_23_layer_call_and_return_conditional_losses_1282438]0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ~
*__inference_dense_23_layer_call_fn_1282447P0в-
&в#
!К
inputs         А
к "К         ╣
G__inference_dropout_21_layer_call_and_return_conditional_losses_1282674n<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ╣
G__inference_dropout_21_layer_call_and_return_conditional_losses_1282686n<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ С
,__inference_dropout_21_layer_call_fn_1282691a<в9
2в/
)К&
inputs         А
p 
к "!К         АС
,__inference_dropout_21_layer_call_fn_1282696a<в9
2в/
)К&
inputs         А
p
к "!К         Ай
G__inference_dropout_22_layer_call_and_return_conditional_losses_1282744^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_22_layer_call_and_return_conditional_losses_1282756^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_22_layer_call_fn_1282761Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_22_layer_call_fn_1282766Q4в1
*в'
!К
inputs         А
p
к "К         Ай
G__inference_dropout_23_layer_call_and_return_conditional_losses_1282803^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_23_layer_call_and_return_conditional_losses_1282815^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_23_layer_call_fn_1282820Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_23_layer_call_fn_1282825Q4в1
*в'
!К
inputs         А
p
к "К         Ам
F__inference_flatten_7_layer_call_and_return_conditional_losses_1282702b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А
Ъ Д
+__inference_flatten_7_layer_call_fn_1282707U8в5
.в+
)К&
inputs         А
к "К         А╣
E__inference_lambda_7_layer_call_and_return_conditional_losses_1282455p?в<
5в2
(К%
inputs         

 
p 
к "-в*
#К 
0         
Ъ ╣
E__inference_lambda_7_layer_call_and_return_conditional_losses_1282463p?в<
5в2
(К%
inputs         

 
p
к "-в*
#К 
0         
Ъ С
*__inference_lambda_7_layer_call_fn_1282468c?в<
5в2
(К%
inputs         

 
p 
к " К         С
*__inference_lambda_7_layer_call_fn_1282473c?в<
5в2
(К%
inputs         

 
p
к " К         <
__inference_loss_fn_0_1282836(в

в 
к "К <
__inference_loss_fn_1_1282847.в

в 
к "К <
__inference_loss_fn_2_12828580в

в 
к "К Ё
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1280156ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_21_layer_call_fn_1280162СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1280168ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_22_layer_call_fn_1280174СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1280180ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_23_layer_call_fn_1280186СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╞
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282004y&'23()*+,-./01?в<
5в2
(К%
inputs         
p 

 
к "&в#
К
0         А
Ъ ╞
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282108y&'23()*+,-./01?в<
5в2
(К%
inputs         
p

 
к "&в#
К
0         А
Ъ ╧
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282191Б&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         
p 

 
к "&в#
К
0         А
Ъ ╧
I__inference_sequential_7_layer_call_and_return_conditional_losses_1282295Б&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         
p

 
к "&в#
К
0         А
Ъ ж
.__inference_sequential_7_layer_call_fn_1282328t&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         
p 

 
к "К         АЮ
.__inference_sequential_7_layer_call_fn_1282361l&'23()*+,-./01?в<
5в2
(К%
inputs         
p 

 
к "К         АЮ
.__inference_sequential_7_layer_call_fn_1282394l&'23()*+,-./01?в<
5в2
(К%
inputs         
p

 
к "К         Аж
.__inference_sequential_7_layer_call_fn_1282427t&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         
p

 
к "К         А╢
%__inference_signature_wrapper_1281353М&'23()*+,-./01Cв@
в 
9к6
4
input_1)К&
input_1         "3к0
.
output_1"К
output_1         