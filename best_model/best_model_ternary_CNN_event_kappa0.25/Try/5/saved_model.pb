»╗
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
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ую
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
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
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
Є
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
ї
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Ё
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
џ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
Њ
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
б
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Џ
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
ё
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: *
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
: *
dtype0
Ё
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*!
shared_nameconv2d_16/kernel
~
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*'
_output_shapes
: ђ*
dtype0
u
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_16/bias
n
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_17/kernel

$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:ђ*
dtype0
}
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђбђ* 
shared_namedense_15/kernel
v
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*!
_output_shapes
:ђбђ*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:ђ*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:ђ*
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
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_17/kernel/m
ѓ
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m
Ћ
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m
Њ
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_15/kernel/m
І
+Adam/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_15/bias/m
{
)Adam/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/m*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_16/kernel/m
ї
+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_16/bias/m
|
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_17/kernel/m
Ї
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_17/bias/m
|
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes	
:ђ*
dtype0
І
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђбђ*'
shared_nameAdam/dense_15/kernel/m
ё
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*!
_output_shapes
:ђбђ*
dtype0
Ђ
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_15/bias/m
z
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_16/kernel/m
Ѓ
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_16/bias/m
z
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_17/kernel/v
ѓ
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v
Ћ
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v
Њ
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_15/kernel/v
І
+Adam/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_15/bias/v
{
)Adam/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/v*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_16/kernel/v
ї
+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_16/bias/v
|
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_17/kernel/v
Ї
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_17/bias/v
|
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes	
:ђ*
dtype0
І
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђбђ*'
shared_nameAdam/dense_15/kernel/v
ё
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*!
_output_shapes
:ђбђ*
dtype0
Ђ
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_15/bias/v
z
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_16/kernel/v
Ѓ
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_16/bias/v
z
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes	
:ђ*
dtype0

NoOpNoOp
У]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Б]
valueЎ]Bќ] BЈ]
і

h2ptjl
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
е
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
п
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m¤'mл*mЛ+mм,mМ-mн.mН/mо0mО1mп2m┘3m┌v█v▄&vП'vя*v▀+vЯ,vр-vР.vс/vС0vт1vТ2vу3vУ
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
Г
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
Ќ
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
Г
rlayer_metrics
	variables
slayer_regularization_losses
tmetrics
unon_trainable_variables
trainable_variables

vlayers
regularization_losses
NL
VARIABLE_VALUEdense_17/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_17/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
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
VARIABLE_VALUEbatch_normalization_5/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_5/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_5/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_5/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_15/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_15/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_16/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_16/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_17/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_17/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_15/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_15/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_16/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_16/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
ђnon_trainable_variables
:regularization_losses
;trainable_variables
Ђlayers
 ѓlayer_regularization_losses
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
Ѓlayer_metrics
>	variables
ёmetrics
Ёnon_trainable_variables
?regularization_losses
@trainable_variables
єlayers
 Єlayer_regularization_losses

*0
+1
 

*0
+1
▓
ѕlayer_metrics
B	variables
Ѕmetrics
іnon_trainable_variables
Cregularization_losses
Dtrainable_variables
Іlayers
 їlayer_regularization_losses
 
 
 
▓
Їlayer_metrics
F	variables
јmetrics
Јnon_trainable_variables
Gregularization_losses
Htrainable_variables
љlayers
 Љlayer_regularization_losses

,0
-1
 

,0
-1
▓
њlayer_metrics
J	variables
Њmetrics
ћnon_trainable_variables
Kregularization_losses
Ltrainable_variables
Ћlayers
 ќlayer_regularization_losses
 
 
 
▓
Ќlayer_metrics
N	variables
ўmetrics
Ўnon_trainable_variables
Oregularization_losses
Ptrainable_variables
џlayers
 Џlayer_regularization_losses

.0
/1
 

.0
/1
▓
юlayer_metrics
R	variables
Юmetrics
ъnon_trainable_variables
Sregularization_losses
Ttrainable_variables
Ъlayers
 аlayer_regularization_losses
 
 
 
▓
Аlayer_metrics
V	variables
бmetrics
Бnon_trainable_variables
Wregularization_losses
Xtrainable_variables
цlayers
 Цlayer_regularization_losses
 
 
 
▓
дlayer_metrics
Z	variables
Дmetrics
еnon_trainable_variables
[regularization_losses
\trainable_variables
Еlayers
 фlayer_regularization_losses
 
 
 
▓
Фlayer_metrics
^	variables
гmetrics
Гnon_trainable_variables
_regularization_losses
`trainable_variables
«layers
 »layer_regularization_losses
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
хlayer_metrics
f	variables
Хmetrics
иnon_trainable_variables
gregularization_losses
htrainable_variables
Иlayers
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
йlayers
 Йlayer_regularization_losses
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
к	variables
К	keras_api
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
к	variables
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
VARIABLE_VALUEAdam/dense_17/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_17/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_15/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_15/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_16/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_16/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_17/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_17/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_15/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_15/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_16/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_16/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_17/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_17/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_15/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_15/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_16/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_16/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_17/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_17/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_15/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_15/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_16/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_16/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
і
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
а
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
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
GPU2 *0J 8ѓ *-
f(R&
$__inference_signature_wrapper_877461
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp+Adam/conv2d_15/kernel/m/Read/ReadVariableOp)Adam/conv2d_15/bias/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp+Adam/conv2d_15/kernel/v/Read/ReadVariableOp)Adam/conv2d_15/bias/v/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOpConst*B
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
GPU2 *0J 8ѓ *(
f#R!
__inference__traced_save_879148
└
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_17/kerneldense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biastotalcounttotal_1count_1Adam/dense_17/kernel/mAdam/dense_17/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/conv2d_15/kernel/mAdam/conv2d_15/bias/mAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/vAdam/dense_17/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/conv2d_15/kernel/vAdam/conv2d_15/bias/vAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/vAdam/dense_16/kernel/vAdam/dense_16/bias/v*A
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
GPU2 *0J 8ѓ *+
f&R$
"__inference__traced_restore_879317ьГ
э
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_878862

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
Ќ
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_876403

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
┬
`
D__inference_lambda_5_layer_call_and_return_conditional_losses_876309

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
Д
Ў
)__inference_dense_16_layer_call_fn_878889

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
D__inference_dense_16_layer_call_and_return_conditional_losses_8764602
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
Ў
Г
$__inference_CNN_layer_call_fn_877535

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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCall»
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_8770742
StatefulPartitionedCallј
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
Д
╝
!__inference__wrapped_model_876132
input_1

cnn_876098:

cnn_876100:

cnn_876102:

cnn_876104:$

cnn_876106: 

cnn_876108: %

cnn_876110: ђ

cnn_876112:	ђ&

cnn_876114:ђђ

cnn_876116:	ђ

cnn_876118:ђбђ

cnn_876120:	ђ

cnn_876122:
ђђ

cnn_876124:	ђ

cnn_876126:	ђ

cnn_876128:
identityѕбCNN/StatefulPartitionedCallЮ
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1
cnn_876098
cnn_876100
cnn_876102
cnn_876104
cnn_876106
cnn_876108
cnn_876110
cnn_876112
cnn_876114
cnn_876116
cnn_876118
cnn_876120
cnn_876122
cnn_876124
cnn_876126
cnn_876128*
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
GPU2 *0J 8ѓ * 
fR
__inference_call_7979872
CNN/StatefulPartitionedCallќ
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
Н
d
+__inference_dropout_17_layer_call_fn_878916

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
F__inference_dropout_17_layer_call_and_return_conditional_losses_8765432
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
м
§
-__inference_sequential_5_layer_call_fn_878128

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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallЏ
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
:         ђ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_8768102
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
х
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_876543

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
Ю
ђ
E__inference_conv2d_16_layer_call_and_return_conditional_losses_876373

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
ш
d
+__inference_dropout_15_layer_call_fn_878787

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
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8766152
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
э
└
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878705

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
┐
Г
D__inference_dense_15_layer_call_and_return_conditional_losses_876430

inputs3
matmul_readvariableop_resource:ђбђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
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
Relu╚
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђб: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         ђб
 
_user_specified_nameinputs
╬
А
*__inference_conv2d_16_layer_call_fn_878746

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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_8763732
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
р
E
)__inference_lambda_5_layer_call_fn_878560

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
D__inference_lambda_5_layer_call_and_return_conditional_losses_8763092
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
╣џ
Ь
H__inference_sequential_5_layer_call_and_return_conditional_losses_878348

inputs;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: C
(conv2d_16_conv2d_readvariableop_resource: ђ8
)conv2d_16_biasadd_readvariableop_resource:	ђD
(conv2d_17_conv2d_readvariableop_resource:ђђ8
)conv2d_17_biasadd_readvariableop_resource:	ђ<
'dense_15_matmul_readvariableop_resource:ђбђ7
(dense_15_biasadd_readvariableop_resource:	ђ;
'dense_16_matmul_readvariableop_resource:
ђђ7
(dense_16_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpбdense_15/BiasAdd/ReadVariableOpбdense_15/MatMul/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpбdense_16/BiasAdd/ReadVariableOpбdense_16/MatMul/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_5/strided_slice/stackЎ
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_5/strided_slice/stack_1Ў
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_5/strided_slice/stack_2ф
lambda_5/strided_sliceStridedSliceinputs%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_5/strided_sliceХ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ж
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3lambda_5/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_5/FusedBatchNormV3░
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue╝
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1│
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOpт
conv2d_15/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_15/Conv2Dф
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp░
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/Relu╩
max_pooling2d_15/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool┤
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_16/Conv2D/ReadVariableOpП
conv2d_16/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_16/Conv2DФ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp▒
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/BiasAdd
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/Relu╦
max_pooling2d_16/MaxPoolMaxPoolconv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPoolх
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_17/Conv2D/ReadVariableOpП
conv2d_17/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_17/Conv2DФ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp▒
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/Relu╦
max_pooling2d_17/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPooly
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_15/dropout/ConstИ
dropout_15/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_15/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_15/dropout/MulЁ
dropout_15/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shapeя
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype021
/dropout_15/dropout/random_uniform/RandomUniformІ
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_15/dropout/GreaterEqual/yз
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2!
dropout_15/dropout/GreaterEqualЕ
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout_15/dropout/Cast»
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout_15/dropout/Mul_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_5/ConstЮ
flatten_5/ReshapeReshapedropout_15/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_5/ReshapeФ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_15/MatMul/ReadVariableOpБ
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/MatMulе
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_15/BiasAdd/ReadVariableOpд
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_15/Reluy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Constф
dropout_16/dropout/MulMuldense_15/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shapeо
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_16/dropout/random_uniform/RandomUniformІ
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/yв
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_16/dropout/GreaterEqualА
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_16/dropout/CastД
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_16/dropout/Mul_1ф
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_16/MatMul/ReadVariableOpЦ
dense_16/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/MatMulе
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_16/BiasAdd/ReadVariableOpд
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_16/Reluy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_17/dropout/Constф
dropout_17/dropout/MulMuldense_16/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shapeо
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_17/dropout/random_uniform/RandomUniformІ
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_17/dropout/GreaterEqual/yв
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_17/dropout/GreaterEqualА
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_17/dropout/CastД
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_17/dropout/Mul_1┘
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulЛ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulл
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulш
IdentityIdentitydropout_17/dropout/Mul_1:z:0%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
И

Ш
D__inference_dense_17_layer_call_and_return_conditional_losses_878555

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
┬
`
D__inference_lambda_5_layer_call_and_return_conditional_losses_878581

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
Ь
Л
6__inference_batch_normalization_5_layer_call_fn_878594

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8761542
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
┘
F
*__inference_flatten_5_layer_call_fn_878809

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
:         ђб* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8764112
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђб2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
э
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_876441

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
х
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_878874

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
┐
└
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878669

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
д
Л
6__inference_batch_normalization_5_layer_call_fn_878620

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8763282
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
Сu
з
__inference_call_800117

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpбBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2в
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЎ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpС
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_5/conv2d_15/BiasAddЦ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_5/conv2d_15/Reluы
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpт
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_5/conv2d_16/BiasAddд
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_5/conv2d_16/ReluЫ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpт
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_5/conv2d_17/BiasAddд
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_5/conv2d_17/ReluЫ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPool╗
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_5/dropout_15/IdentityЇ
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/ConstЛ
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpО
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp┌
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/BiasAddЏ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/ReluГ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_16/IdentityЛ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp┘
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp┌
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/BiasAddЏ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/ReluГ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_17/IdentityЕ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOp▒
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЦ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_17/Softmax■
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
І
ю
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878651

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
У
│
__inference_loss_fn_2_878966N
:dense_16_kernel_regularizer_square_readvariableop_resource:
ђђ
identityѕб1dense_16/kernel/Regularizer/Square/ReadVariableOpс
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_16_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulџ
IdentityIdentity#dense_16/kernel/Regularizer/mul:z:02^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp
г
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_876264

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
н
§
-__inference_sequential_5_layer_call_fn_878095

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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallЮ
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
:         ђ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_8764922
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
Ё\
і
H__inference_sequential_5_layer_call_and_return_conditional_losses_876492

inputs*
batch_normalization_5_876329:*
batch_normalization_5_876331:*
batch_normalization_5_876333:*
batch_normalization_5_876335:*
conv2d_15_876356: 
conv2d_15_876358: +
conv2d_16_876374: ђ
conv2d_16_876376:	ђ,
conv2d_17_876392:ђђ
conv2d_17_876394:	ђ$
dense_15_876431:ђбђ
dense_15_876433:	ђ#
dense_16_876461:
ђђ
dense_16_876463:	ђ
identityѕб-batch_normalization_5/StatefulPartitionedCallб!conv2d_15/StatefulPartitionedCallб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб!conv2d_16/StatefulPartitionedCallб!conv2d_17/StatefulPartitionedCallб dense_15/StatefulPartitionedCallб1dense_15/kernel/Regularizer/Square/ReadVariableOpб dense_16/StatefulPartitionedCallб1dense_16/kernel/Regularizer/Square/ReadVariableOpр
lambda_5/PartitionedCallPartitionedCallinputs*
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
D__inference_lambda_5_layer_call_and_return_conditional_losses_8763092
lambda_5/PartitionedCallй
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0batch_normalization_5_876329batch_normalization_5_876331batch_normalization_5_876333batch_normalization_5_876335*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8763282/
-batch_normalization_5/StatefulPartitionedCallо
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_15_876356conv2d_15_876358*
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
E__inference_conv2d_15_layer_call_and_return_conditional_losses_8763552#
!conv2d_15/StatefulPartitionedCallЮ
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_8762642"
 max_pooling2d_15/PartitionedCall╩
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_16_876374conv2d_16_876376*
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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_8763732#
!conv2d_16/StatefulPartitionedCallъ
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_8762762"
 max_pooling2d_16/PartitionedCall╩
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_17_876392conv2d_17_876394*
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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_8763912#
!conv2d_17/StatefulPartitionedCallъ
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_8762882"
 max_pooling2d_17/PartitionedCallІ
dropout_15/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8764032
dropout_15/PartitionedCallч
flatten_5/PartitionedCallPartitionedCall#dropout_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђб* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8764112
flatten_5/PartitionedCallХ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_876431dense_15_876433*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_8764302"
 dense_15/StatefulPartitionedCallЃ
dropout_16/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_8764412
dropout_16/PartitionedCallи
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_876461dense_16_876463*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_8764602"
 dense_16/StatefulPartitionedCallЃ
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_8764712
dropout_17/PartitionedCall┴
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_876356*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul╣
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_876431*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulИ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_876461* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulэ
IdentityIdentity#dropout_17/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ш
e
F__inference_dropout_15_layer_call_and_return_conditional_losses_876615

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
:         		ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
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
:         		ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
ф
џ
)__inference_dense_15_layer_call_fn_878830

inputs
unknown:ђбђ
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
D__inference_dense_15_layer_call_and_return_conditional_losses_8764302
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђб: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ђб
 
_user_specified_nameinputs
─▀
╠!
"__inference__traced_restore_879317
file_prefix3
 assignvariableop_dense_17_kernel:	ђ.
 assignvariableop_1_dense_17_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_5_gamma:;
-assignvariableop_8_batch_normalization_5_beta:B
4assignvariableop_9_batch_normalization_5_moving_mean:G
9assignvariableop_10_batch_normalization_5_moving_variance:>
$assignvariableop_11_conv2d_15_kernel: 0
"assignvariableop_12_conv2d_15_bias: ?
$assignvariableop_13_conv2d_16_kernel: ђ1
"assignvariableop_14_conv2d_16_bias:	ђ@
$assignvariableop_15_conv2d_17_kernel:ђђ1
"assignvariableop_16_conv2d_17_bias:	ђ8
#assignvariableop_17_dense_15_kernel:ђбђ0
!assignvariableop_18_dense_15_bias:	ђ7
#assignvariableop_19_dense_16_kernel:
ђђ0
!assignvariableop_20_dense_16_bias:	ђ#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_17_kernel_m:	ђ6
(assignvariableop_26_adam_dense_17_bias_m:D
6assignvariableop_27_adam_batch_normalization_5_gamma_m:C
5assignvariableop_28_adam_batch_normalization_5_beta_m:E
+assignvariableop_29_adam_conv2d_15_kernel_m: 7
)assignvariableop_30_adam_conv2d_15_bias_m: F
+assignvariableop_31_adam_conv2d_16_kernel_m: ђ8
)assignvariableop_32_adam_conv2d_16_bias_m:	ђG
+assignvariableop_33_adam_conv2d_17_kernel_m:ђђ8
)assignvariableop_34_adam_conv2d_17_bias_m:	ђ?
*assignvariableop_35_adam_dense_15_kernel_m:ђбђ7
(assignvariableop_36_adam_dense_15_bias_m:	ђ>
*assignvariableop_37_adam_dense_16_kernel_m:
ђђ7
(assignvariableop_38_adam_dense_16_bias_m:	ђ=
*assignvariableop_39_adam_dense_17_kernel_v:	ђ6
(assignvariableop_40_adam_dense_17_bias_v:D
6assignvariableop_41_adam_batch_normalization_5_gamma_v:C
5assignvariableop_42_adam_batch_normalization_5_beta_v:E
+assignvariableop_43_adam_conv2d_15_kernel_v: 7
)assignvariableop_44_adam_conv2d_15_bias_v: F
+assignvariableop_45_adam_conv2d_16_kernel_v: ђ8
)assignvariableop_46_adam_conv2d_16_bias_v:	ђG
+assignvariableop_47_adam_conv2d_17_kernel_v:ђђ8
)assignvariableop_48_adam_conv2d_17_bias_v:	ђ?
*assignvariableop_49_adam_dense_15_kernel_v:ђбђ7
(assignvariableop_50_adam_dense_15_bias_v:	ђ>
*assignvariableop_51_adam_dense_16_kernel_v:
ђђ7
(assignvariableop_52_adam_dense_16_bias_v:	ђ
identity_54ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9В
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Э
valueЬBв6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЩ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ь
_output_shapes█
п::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_17_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_17_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_5_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_5_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╣
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_5_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┴
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_5_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_15_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ф
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_15_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_16_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ф
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_16_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_17_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ф
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_17_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ф
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_15_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Е
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_15_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ф
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_16_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_16_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21А
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Б
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
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
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_17_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_17_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Й
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_5_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28й
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_5_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_15_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_15_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_16_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_16_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_17_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_17_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_15_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36░
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_15_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_16_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_16_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_17_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_17_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Й
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_5_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42й
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_5_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_15_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_15_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_16_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_16_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_17_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_17_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▓
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_15_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50░
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_15_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▓
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_16_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52░
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_16_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpВ	
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
╣
г
D__inference_dense_16_layer_call_and_return_conditional_losses_876460

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpЈ
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
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
х
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_876576

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
Ќ
Г
$__inference_CNN_layer_call_fn_877572

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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCallГ
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_8772142
StatefulPartitionedCallј
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
Ќ
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_878792

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
Сu
з
__inference_call_797987

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpбBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2в
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЎ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpС
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_5/conv2d_15/BiasAddЦ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_5/conv2d_15/Reluы
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpт
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_5/conv2d_16/BiasAddд
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_5/conv2d_16/ReluЫ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpт
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_5/conv2d_17/BiasAddд
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_5/conv2d_17/ReluЫ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPool╗
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_5/dropout_15/IdentityЇ
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/ConstЛ
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpО
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp┌
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/BiasAddЏ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/ReluГ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_16/IdentityЛ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp┘
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp┌
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/BiasAddЏ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/ReluГ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_17/IdentityЕ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOp▒
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЦ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_17/Softmax■
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╔
G
+__inference_dropout_17_layer_call_fn_878911

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
F__inference_dropout_17_layer_call_and_return_conditional_losses_8764712
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
§ћ
И
?__inference_CNN_layer_call_and_return_conditional_losses_877699

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpбBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2в
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЎ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpС
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_5/conv2d_15/BiasAddЦ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_5/conv2d_15/Reluы
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpт
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_5/conv2d_16/BiasAddд
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_5/conv2d_16/ReluЫ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpт
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_5/conv2d_17/BiasAddд
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_5/conv2d_17/ReluЫ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPool╗
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_5/dropout_15/IdentityЇ
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/ConstЛ
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpО
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp┌
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/BiasAddЏ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/ReluГ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_16/IdentityЛ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp┘
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp┌
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/BiasAddЏ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/ReluГ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_17/IdentityЕ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOp▒
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЦ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_17/SoftmaxТ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulя
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulП
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЏ
IdentityIdentitydense_17/Softmax:softmax:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ж
Ё
-__inference_sequential_5_layer_call_fn_878161
lambda_5_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_8768102
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
_user_specified_namelambda_5_input
г
h
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_876288

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
џ
«
$__inference_CNN_layer_call_fn_877609
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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCall«
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_8772142
StatefulPartitionedCallј
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
В
Л
6__inference_batch_normalization_5_layer_call_fn_878607

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8761982
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
Ю
ђ
E__inference_conv2d_16_layer_call_and_return_conditional_losses_878757

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
┬
`
D__inference_lambda_5_layer_call_and_return_conditional_losses_878573

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
І
ю
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_876154

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
ж
G
+__inference_dropout_15_layer_call_fn_878782

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
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8764032
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
Н
d
+__inference_dropout_16_layer_call_fn_878857

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
F__inference_dropout_16_layer_call_and_return_conditional_losses_8765762
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
Зs
з
__inference_call_799973

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpбBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2с
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1║
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp▄
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_5/conv2d_15/BiasAddЮ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_5/conv2d_15/Reluж
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЅ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpП
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_5/conv2d_16/BiasAddъ
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_5/conv2d_16/ReluЖ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЅ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpП
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_5/conv2d_17/BiasAddъ
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_5/conv2d_17/ReluЖ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPool│
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*(
_output_shapes
:ђ		ђ2"
 sequential_5/dropout_15/IdentityЇ
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/Const╔
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*!
_output_shapes
:ђђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOp¤
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOpм
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_15/BiasAddЊ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_15/ReluЦ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_5/dropout_16/IdentityЛ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOpЛ
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpм
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_16/BiasAddЊ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_16/ReluЦ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_5/dropout_17/IdentityЕ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOpЕ
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЮ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_17/BiasAddt
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_17/SoftmaxШ
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ђKK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_876276

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
в
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_878815

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
:         ђб2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђб2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
Б
Ќ
)__inference_dense_17_layer_call_fn_878544

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
D__inference_dense_17_layer_call_and_return_conditional_losses_8770492
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
я
M
1__inference_max_pooling2d_16_layer_call_fn_876282

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
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_8762762
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
ц
Л
6__inference_batch_normalization_5_layer_call_fn_878633

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8766812
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
я
M
1__inference_max_pooling2d_17_layer_call_fn_876294

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
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_8762882
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
ђЋ
╣
?__inference_CNN_layer_call_and_return_conditional_losses_877900
input_1H
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpбBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2В
#sequential_5/lambda_5/strided_sliceStridedSliceinput_12sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЎ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpС
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_5/conv2d_15/BiasAddЦ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_5/conv2d_15/Reluы
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpт
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_5/conv2d_16/BiasAddд
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_5/conv2d_16/ReluЫ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpт
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_5/conv2d_17/BiasAddд
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_5/conv2d_17/ReluЫ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPool╗
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_5/dropout_15/IdentityЇ
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/ConstЛ
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpО
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp┌
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/BiasAddЏ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/ReluГ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_16/IdentityЛ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp┘
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp┌
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/BiasAddЏ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/ReluГ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_5/dropout_17/IdentityЕ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOp▒
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЦ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_17/SoftmaxТ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulя
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulП
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЏ
IdentityIdentitydense_17/Softmax:softmax:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Аv
д
H__inference_sequential_5_layer_call_and_return_conditional_losses_878431
lambda_5_input;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: C
(conv2d_16_conv2d_readvariableop_resource: ђ8
)conv2d_16_biasadd_readvariableop_resource:	ђD
(conv2d_17_conv2d_readvariableop_resource:ђђ8
)conv2d_17_biasadd_readvariableop_resource:	ђ<
'dense_15_matmul_readvariableop_resource:ђбђ7
(dense_15_biasadd_readvariableop_resource:	ђ;
'dense_16_matmul_readvariableop_resource:
ђђ7
(dense_16_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpбdense_15/BiasAdd/ReadVariableOpбdense_15/MatMul/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpбdense_16/BiasAdd/ReadVariableOpбdense_16/MatMul/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_5/strided_slice/stackЎ
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_5/strided_slice/stack_1Ў
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_5/strided_slice/stack_2▓
lambda_5/strided_sliceStridedSlicelambda_5_input%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_5/strided_sliceХ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ж
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3lambda_5/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3│
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOpт
conv2d_15/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_15/Conv2Dф
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp░
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/Relu╩
max_pooling2d_15/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool┤
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_16/Conv2D/ReadVariableOpП
conv2d_16/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_16/Conv2DФ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp▒
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/BiasAdd
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/Relu╦
max_pooling2d_16/MaxPoolMaxPoolconv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPoolх
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_17/Conv2D/ReadVariableOpП
conv2d_17/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_17/Conv2DФ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp▒
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/Relu╦
max_pooling2d_17/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPoolћ
dropout_15/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_15/Identitys
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_5/ConstЮ
flatten_5/ReshapeReshapedropout_15/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_5/ReshapeФ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_15/MatMul/ReadVariableOpБ
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/MatMulе
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_15/BiasAdd/ReadVariableOpд
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_15/Reluє
dropout_16/IdentityIdentitydense_15/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_16/Identityф
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_16/MatMul/ReadVariableOpЦ
dense_16/MatMulMatMuldropout_16/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/MatMulе
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_16/BiasAdd/ReadVariableOpд
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_16/Reluє
dropout_17/IdentityIdentitydense_16/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_17/Identity┘
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulЛ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulл
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЦ
IdentityIdentitydropout_17/Identity:output:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_5_input
э
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_878921

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
џ
╗
__inference_loss_fn_0_878944U
;conv2d_15_kernel_regularizer_square_readvariableop_resource: 
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpВ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_15_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulю
IdentityIdentity$conv2d_15/kernel/Regularizer/mul:z:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp
╔
G
+__inference_dropout_16_layer_call_fn_878852

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
F__inference_dropout_16_layer_call_and_return_conditional_losses_8764412
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
А
Ђ
E__inference_conv2d_17_layer_call_and_return_conditional_losses_878777

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
э
└
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_876681

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
┐
│
E__inference_conv2d_15_layer_call_and_return_conditional_losses_878737

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpЋ
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
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐
│
E__inference_conv2d_15_layer_call_and_return_conditional_losses_876355

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpЋ
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
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╩
Ъ
*__inference_conv2d_15_layer_call_fn_878720

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
E__inference_conv2d_15_layer_call_and_return_conditional_losses_8763552
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
р
E
)__inference_lambda_5_layer_call_fn_878565

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
D__inference_lambda_5_layer_call_and_return_conditional_losses_8767082
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
╣
г
D__inference_dense_16_layer_call_and_return_conditional_losses_878906

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpЈ
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
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
в
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_876411

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
:         ђб2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђб2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
Ѕv
ъ
H__inference_sequential_5_layer_call_and_return_conditional_losses_878244

inputs;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: C
(conv2d_16_conv2d_readvariableop_resource: ђ8
)conv2d_16_biasadd_readvariableop_resource:	ђD
(conv2d_17_conv2d_readvariableop_resource:ђђ8
)conv2d_17_biasadd_readvariableop_resource:	ђ<
'dense_15_matmul_readvariableop_resource:ђбђ7
(dense_15_biasadd_readvariableop_resource:	ђ;
'dense_16_matmul_readvariableop_resource:
ђђ7
(dense_16_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpбdense_15/BiasAdd/ReadVariableOpбdense_15/MatMul/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpбdense_16/BiasAdd/ReadVariableOpбdense_16/MatMul/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_5/strided_slice/stackЎ
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_5/strided_slice/stack_1Ў
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_5/strided_slice/stack_2ф
lambda_5/strided_sliceStridedSliceinputs%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_5/strided_sliceХ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ж
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3lambda_5/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3│
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOpт
conv2d_15/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_15/Conv2Dф
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp░
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/Relu╩
max_pooling2d_15/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool┤
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_16/Conv2D/ReadVariableOpП
conv2d_16/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_16/Conv2DФ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp▒
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/BiasAdd
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/Relu╦
max_pooling2d_16/MaxPoolMaxPoolconv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPoolх
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_17/Conv2D/ReadVariableOpП
conv2d_17/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_17/Conv2DФ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp▒
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/Relu╦
max_pooling2d_17/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPoolћ
dropout_15/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_15/Identitys
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_5/ConstЮ
flatten_5/ReshapeReshapedropout_15/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_5/ReshapeФ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_15/MatMul/ReadVariableOpБ
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/MatMulе
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_15/BiasAdd/ReadVariableOpд
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_15/Reluє
dropout_16/IdentityIdentitydense_15/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_16/Identityф
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_16/MatMul/ReadVariableOpЦ
dense_16/MatMulMatMuldropout_16/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/MatMulе
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_16/BiasAdd/ReadVariableOpд
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_16/Reluє
dropout_17/IdentityIdentitydense_16/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_17/Identity┘
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulЛ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulл
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЦ
IdentityIdentitydropout_17/Identity:output:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐
Г
D__inference_dense_15_layer_call_and_return_conditional_losses_878847

inputs3
matmul_readvariableop_resource:ђбђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
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
Relu╚
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђб: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         ђб
 
_user_specified_nameinputs
ю
«
$__inference_CNN_layer_call_fn_877498
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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCall░
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_8770742
StatefulPartitionedCallј
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
љh
К
__inference__traced_save_879148
file_prefix.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop6
2savev2_adam_conv2d_15_kernel_m_read_readvariableop4
0savev2_adam_conv2d_15_bias_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop6
2savev2_adam_conv2d_15_kernel_v_read_readvariableop4
0savev2_adam_conv2d_15_bias_v_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop
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
ShardedFilenameТ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Э
valueЬBв6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesШ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop2savev2_adam_conv2d_15_kernel_m_read_readvariableop0savev2_adam_conv2d_15_bias_m_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop2savev2_adam_conv2d_15_kernel_v_read_readvariableop0savev2_adam_conv2d_15_bias_v_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*Ж
_input_shapesп
Н: :	ђ:: : : : : ::::: : : ђ:ђ:ђђ:ђ:ђбђ:ђ:
ђђ:ђ: : : : :	ђ:::: : : ђ:ђ:ђђ:ђ:ђбђ:ђ:
ђђ:ђ:	ђ:::: : : ђ:ђ:ђђ:ђ:ђбђ:ђ:
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
: ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:'#
!
_output_shapes
:ђбђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:
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
:	ђ: 
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
: ђ:!!

_output_shapes	
:ђ:."*
(
_output_shapes
:ђђ:!#

_output_shapes	
:ђ:'$#
!
_output_shapes
:ђбђ:!%

_output_shapes	
:ђ:&&"
 
_output_shapes
:
ђђ:!'

_output_shapes	
:ђ:%(!

_output_shapes
:	ђ: )
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
: ђ:!/

_output_shapes	
:ђ:.0*
(
_output_shapes
:ђђ:!1

_output_shapes	
:ђ:'2#
!
_output_shapes
:ђбђ:!3

_output_shapes	
:ђ:&4"
 
_output_shapes
:
ђђ:!5

_output_shapes	
:ђ:6

_output_shapes
: 
Ш┴
б
?__inference_CNN_layer_call_and_return_conditional_losses_877810

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpб1sequential_5/batch_normalization_5/AssignNewValueб3sequential_5/batch_normalization_5/AssignNewValue_1бBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2в
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1л
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_5/batch_normalization_5/FusedBatchNormV3ы
1sequential_5/batch_normalization_5/AssignNewValueAssignVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource@sequential_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_5/batch_normalization_5/AssignNewValue§
3sequential_5/batch_normalization_5/AssignNewValue_1AssignVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceDsequential_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0E^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_5/batch_normalization_5/AssignNewValue_1┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЎ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpС
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_5/conv2d_15/BiasAddЦ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_5/conv2d_15/Reluы
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpт
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_5/conv2d_16/BiasAddд
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_5/conv2d_16/ReluЫ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpт
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_5/conv2d_17/BiasAddд
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_5/conv2d_17/ReluЫ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPoolЊ
%sequential_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_5/dropout_15/dropout/ConstВ
#sequential_5/dropout_15/dropout/MulMul.sequential_5/max_pooling2d_17/MaxPool:output:0.sequential_5/dropout_15/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2%
#sequential_5/dropout_15/dropout/Mulг
%sequential_5/dropout_15/dropout/ShapeShape.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_5/dropout_15/dropout/ShapeЁ
<sequential_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_15/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype02>
<sequential_5/dropout_15/dropout/random_uniform/RandomUniformЦ
.sequential_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_5/dropout_15/dropout/GreaterEqual/yД
,sequential_5/dropout_15/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_15/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2.
,sequential_5/dropout_15/dropout/GreaterEqualл
$sequential_5/dropout_15/dropout/CastCast0sequential_5/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2&
$sequential_5/dropout_15/dropout/Castс
%sequential_5/dropout_15/dropout/Mul_1Mul'sequential_5/dropout_15/dropout/Mul:z:0(sequential_5/dropout_15/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2'
%sequential_5/dropout_15/dropout/Mul_1Ї
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/ConstЛ
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/dropout/Mul_1:z:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpО
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp┌
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/BiasAddЏ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/ReluЊ
%sequential_5/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_5/dropout_16/dropout/Constя
#sequential_5/dropout_16/dropout/MulMul(sequential_5/dense_15/Relu:activations:0.sequential_5/dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_5/dropout_16/dropout/Mulд
%sequential_5/dropout_16/dropout/ShapeShape(sequential_5/dense_15/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dropout_16/dropout/Shape§
<sequential_5/dropout_16/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_5/dropout_16/dropout/random_uniform/RandomUniformЦ
.sequential_5/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_5/dropout_16/dropout/GreaterEqual/yЪ
,sequential_5/dropout_16/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_16/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_5/dropout_16/dropout/GreaterEqual╚
$sequential_5/dropout_16/dropout/CastCast0sequential_5/dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_5/dropout_16/dropout/Cast█
%sequential_5/dropout_16/dropout/Mul_1Mul'sequential_5/dropout_16/dropout/Mul:z:0(sequential_5/dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_5/dropout_16/dropout/Mul_1Л
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp┘
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/dropout/Mul_1:z:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp┌
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/BiasAddЏ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/ReluЊ
%sequential_5/dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_5/dropout_17/dropout/Constя
#sequential_5/dropout_17/dropout/MulMul(sequential_5/dense_16/Relu:activations:0.sequential_5/dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_5/dropout_17/dropout/Mulд
%sequential_5/dropout_17/dropout/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dropout_17/dropout/Shape§
<sequential_5/dropout_17/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_5/dropout_17/dropout/random_uniform/RandomUniformЦ
.sequential_5/dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_5/dropout_17/dropout/GreaterEqual/yЪ
,sequential_5/dropout_17/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_17/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_5/dropout_17/dropout/GreaterEqual╚
$sequential_5/dropout_17/dropout/CastCast0sequential_5/dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_5/dropout_17/dropout/Cast█
%sequential_5/dropout_17/dropout/Mul_1Mul'sequential_5/dropout_17/dropout/Mul:z:0(sequential_5/dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_5/dropout_17/dropout/Mul_1Е
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOp▒
dense_17/MatMulMatMul)sequential_5/dropout_17/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЦ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_17/SoftmaxТ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulя
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulП
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЁ	
IdentityIdentitydense_17/Softmax:softmax:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^sequential_5/batch_normalization_5/AssignNewValue4^sequential_5/batch_normalization_5/AssignNewValue_1C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1sequential_5/batch_normalization_5/AssignNewValue1sequential_5/batch_normalization_5/AssignNewValue2j
3sequential_5/batch_normalization_5/AssignNewValue_13sequential_5/batch_normalization_5/AssignNewValue_12ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
■
«
$__inference_signature_wrapper_877461
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
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCallњ
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
GPU2 *0J 8ѓ **
f%R#
!__inference__wrapped_model_8761322
StatefulPartitionedCallј
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
┬
`
D__inference_lambda_5_layer_call_and_return_conditional_losses_876708

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
Зs
з
__inference_call_800045

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpбBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2с
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1║
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp▄
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_5/conv2d_15/BiasAddЮ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_5/conv2d_15/Reluж
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЅ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpП
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_5/conv2d_16/BiasAddъ
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_5/conv2d_16/ReluЖ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЅ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpП
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_5/conv2d_17/BiasAddъ
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_5/conv2d_17/ReluЖ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPool│
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*(
_output_shapes
:ђ		ђ2"
 sequential_5/dropout_15/IdentityЇ
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/Const╔
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*!
_output_shapes
:ђђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOp¤
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOpм
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_15/BiasAddЊ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_15/ReluЦ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_5/dropout_16/IdentityЛ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOpЛ
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpм
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_16/BiasAddЊ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_5/dense_16/ReluЦ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_5/dropout_17/IdentityЕ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOpЕ
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЮ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_17/BiasAddt
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_17/SoftmaxШ
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ђKK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
Л
б
*__inference_conv2d_17_layer_call_fn_878766

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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_8763912
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
├
ю
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878687

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
А
Ђ
E__inference_conv2d_17_layer_call_and_return_conditional_losses_876391

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
ш1
ф
?__inference_CNN_layer_call_and_return_conditional_losses_877074

inputs!
sequential_5_877009:!
sequential_5_877011:!
sequential_5_877013:!
sequential_5_877015:-
sequential_5_877017: !
sequential_5_877019: .
sequential_5_877021: ђ"
sequential_5_877023:	ђ/
sequential_5_877025:ђђ"
sequential_5_877027:	ђ(
sequential_5_877029:ђбђ"
sequential_5_877031:	ђ'
sequential_5_877033:
ђђ"
sequential_5_877035:	ђ"
dense_17_877050:	ђ
dense_17_877052:
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpб dense_17/StatefulPartitionedCallб$sequential_5/StatefulPartitionedCall┬
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_877009sequential_5_877011sequential_5_877013sequential_5_877015sequential_5_877017sequential_5_877019sequential_5_877021sequential_5_877023sequential_5_877025sequential_5_877027sequential_5_877029sequential_5_877031sequential_5_877033sequential_5_877035*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_8764922&
$sequential_5/StatefulPartitionedCall└
 dense_17/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0dense_17_877050dense_17_877052*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_8770492"
 dense_17/StatefulPartitionedCall─
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_877017*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulй
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_877029*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul╝
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_877033* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulС
IdentityIdentity)dense_17/StatefulPartitionedCall:output:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_15_layer_call_fn_876270

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
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_8762642
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
ж`
щ
H__inference_sequential_5_layer_call_and_return_conditional_losses_876810

inputs*
batch_normalization_5_876750:*
batch_normalization_5_876752:*
batch_normalization_5_876754:*
batch_normalization_5_876756:*
conv2d_15_876759: 
conv2d_15_876761: +
conv2d_16_876765: ђ
conv2d_16_876767:	ђ,
conv2d_17_876771:ђђ
conv2d_17_876773:	ђ$
dense_15_876779:ђбђ
dense_15_876781:	ђ#
dense_16_876785:
ђђ
dense_16_876787:	ђ
identityѕб-batch_normalization_5/StatefulPartitionedCallб!conv2d_15/StatefulPartitionedCallб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб!conv2d_16/StatefulPartitionedCallб!conv2d_17/StatefulPartitionedCallб dense_15/StatefulPartitionedCallб1dense_15/kernel/Regularizer/Square/ReadVariableOpб dense_16/StatefulPartitionedCallб1dense_16/kernel/Regularizer/Square/ReadVariableOpб"dropout_15/StatefulPartitionedCallб"dropout_16/StatefulPartitionedCallб"dropout_17/StatefulPartitionedCallр
lambda_5/PartitionedCallPartitionedCallinputs*
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
D__inference_lambda_5_layer_call_and_return_conditional_losses_8767082
lambda_5/PartitionedCall╗
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0batch_normalization_5_876750batch_normalization_5_876752batch_normalization_5_876754batch_normalization_5_876756*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8766812/
-batch_normalization_5/StatefulPartitionedCallо
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_15_876759conv2d_15_876761*
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
E__inference_conv2d_15_layer_call_and_return_conditional_losses_8763552#
!conv2d_15/StatefulPartitionedCallЮ
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_8762642"
 max_pooling2d_15/PartitionedCall╩
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_16_876765conv2d_16_876767*
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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_8763732#
!conv2d_16/StatefulPartitionedCallъ
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_8762762"
 max_pooling2d_16/PartitionedCall╩
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_17_876771conv2d_17_876773*
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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_8763912#
!conv2d_17/StatefulPartitionedCallъ
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_8762882"
 max_pooling2d_17/PartitionedCallБ
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8766152$
"dropout_15/StatefulPartitionedCallЃ
flatten_5/PartitionedCallPartitionedCall+dropout_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђб* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8764112
flatten_5/PartitionedCallХ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_876779dense_15_876781*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_8764302"
 dense_15/StatefulPartitionedCall└
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_8765762$
"dropout_16/StatefulPartitionedCall┐
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_876785dense_16_876787*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_8764602"
 dense_16/StatefulPartitionedCall└
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_8765432$
"dropout_17/StatefulPartitionedCall┴
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_876759*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul╣
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_876779*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulИ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_876785* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЬ
IdentityIdentity+dropout_17/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
х
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_878933

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
в
┤
__inference_loss_fn_1_878955O
:dense_15_kernel_regularizer_square_readvariableop_resource:ђбђ
identityѕб1dense_15/kernel/Regularizer/Square/ReadVariableOpС
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_15_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulџ
IdentityIdentity#dense_15/kernel/Regularizer/mul:z:02^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp
Ш
e
F__inference_dropout_15_layer_call_and_return_conditional_losses_878804

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
:         		ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
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
:         		ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
В
Ё
-__inference_sequential_5_layer_call_fn_878062
lambda_5_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_8764922
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
_user_specified_namelambda_5_input
├
ю
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_876328

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
D__inference_dense_17_layer_call_and_return_conditional_losses_877049

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
з1
ф
?__inference_CNN_layer_call_and_return_conditional_losses_877214

inputs!
sequential_5_877161:!
sequential_5_877163:!
sequential_5_877165:!
sequential_5_877167:-
sequential_5_877169: !
sequential_5_877171: .
sequential_5_877173: ђ"
sequential_5_877175:	ђ/
sequential_5_877177:ђђ"
sequential_5_877179:	ђ(
sequential_5_877181:ђбђ"
sequential_5_877183:	ђ'
sequential_5_877185:
ђђ"
sequential_5_877187:	ђ"
dense_17_877190:	ђ
dense_17_877192:
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpб dense_17/StatefulPartitionedCallб$sequential_5/StatefulPartitionedCall└
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_877161sequential_5_877163sequential_5_877165sequential_5_877167sequential_5_877169sequential_5_877171sequential_5_877173sequential_5_877175sequential_5_877177sequential_5_877179sequential_5_877181sequential_5_877183sequential_5_877185sequential_5_877187*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_8768102&
$sequential_5/StatefulPartitionedCall└
 dense_17/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0dense_17_877190dense_17_877192*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_8770492"
 dense_17/StatefulPartitionedCall─
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_877169*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulй
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_877181*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul╝
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_877185* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulС
IdentityIdentity)dense_17/StatefulPartitionedCall:output:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
э
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_876471

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
щ┴
Б
?__inference_CNN_layer_call_and_return_conditional_losses_878011
input_1H
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: ђE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	ђQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ђђE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	ђI
4sequential_5_dense_15_matmul_readvariableop_resource:ђбђD
5sequential_5_dense_15_biasadd_readvariableop_resource:	ђH
4sequential_5_dense_16_matmul_readvariableop_resource:
ђђD
5sequential_5_dense_16_biasadd_readvariableop_resource:	ђ:
'dense_17_matmul_readvariableop_resource:	ђ6
(dense_17_biasadd_readvariableop_resource:
identityѕб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpбdense_17/BiasAdd/ReadVariableOpбdense_17/MatMul/ReadVariableOpб1sequential_5/batch_normalization_5/AssignNewValueб3sequential_5/batch_normalization_5/AssignNewValue_1бBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_5/batch_normalization_5/ReadVariableOpб3sequential_5/batch_normalization_5/ReadVariableOp_1б-sequential_5/conv2d_15/BiasAdd/ReadVariableOpб,sequential_5/conv2d_15/Conv2D/ReadVariableOpб-sequential_5/conv2d_16/BiasAdd/ReadVariableOpб,sequential_5/conv2d_16/Conv2D/ReadVariableOpб-sequential_5/conv2d_17/BiasAdd/ReadVariableOpб,sequential_5/conv2d_17/Conv2D/ReadVariableOpб,sequential_5/dense_15/BiasAdd/ReadVariableOpб+sequential_5/dense_15/MatMul/ReadVariableOpб,sequential_5/dense_16/BiasAdd/ReadVariableOpб+sequential_5/dense_16/MatMul/ReadVariableOp»
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_5/lambda_5/strided_slice/stack│
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_5/lambda_5/strided_slice/stack_1│
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_5/lambda_5/strided_slice/stack_2В
#sequential_5/lambda_5/strided_sliceStridedSliceinput_12sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_5/lambda_5/strided_sliceП
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpс
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1љ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpќ
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1л
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_5/batch_normalization_5/FusedBatchNormV3ы
1sequential_5/batch_normalization_5/AssignNewValueAssignVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource@sequential_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_5/batch_normalization_5/AssignNewValue§
3sequential_5/batch_normalization_5/AssignNewValue_1AssignVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceDsequential_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0E^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_5/batch_normalization_5/AssignNewValue_1┌
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_5/conv2d_15/Conv2D/ReadVariableOpЎ
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_5/conv2d_15/Conv2DЛ
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpС
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_5/conv2d_15/BiasAddЦ
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_5/conv2d_15/Reluы
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_15/MaxPool█
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_5/conv2d_16/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_5/conv2d_16/Conv2Dм
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpт
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_5/conv2d_16/BiasAddд
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_5/conv2d_16/ReluЫ
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_16/MaxPool▄
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_5/conv2d_17/Conv2D/ReadVariableOpЉ
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_5/conv2d_17/Conv2Dм
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpт
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_5/conv2d_17/BiasAddд
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_5/conv2d_17/ReluЫ
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_5/max_pooling2d_17/MaxPoolЊ
%sequential_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_5/dropout_15/dropout/ConstВ
#sequential_5/dropout_15/dropout/MulMul.sequential_5/max_pooling2d_17/MaxPool:output:0.sequential_5/dropout_15/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2%
#sequential_5/dropout_15/dropout/Mulг
%sequential_5/dropout_15/dropout/ShapeShape.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_5/dropout_15/dropout/ShapeЁ
<sequential_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_15/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype02>
<sequential_5/dropout_15/dropout/random_uniform/RandomUniformЦ
.sequential_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_5/dropout_15/dropout/GreaterEqual/yД
,sequential_5/dropout_15/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_15/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2.
,sequential_5/dropout_15/dropout/GreaterEqualл
$sequential_5/dropout_15/dropout/CastCast0sequential_5/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2&
$sequential_5/dropout_15/dropout/Castс
%sequential_5/dropout_15/dropout/Mul_1Mul'sequential_5/dropout_15/dropout/Mul:z:0(sequential_5/dropout_15/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2'
%sequential_5/dropout_15/dropout/Mul_1Ї
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_5/flatten_5/ConstЛ
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/dropout/Mul_1:z:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_5/flatten_5/Reshapeм
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpО
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/MatMul¤
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp┌
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/BiasAddЏ
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_15/ReluЊ
%sequential_5/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_5/dropout_16/dropout/Constя
#sequential_5/dropout_16/dropout/MulMul(sequential_5/dense_15/Relu:activations:0.sequential_5/dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_5/dropout_16/dropout/Mulд
%sequential_5/dropout_16/dropout/ShapeShape(sequential_5/dense_15/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dropout_16/dropout/Shape§
<sequential_5/dropout_16/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_5/dropout_16/dropout/random_uniform/RandomUniformЦ
.sequential_5/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_5/dropout_16/dropout/GreaterEqual/yЪ
,sequential_5/dropout_16/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_16/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_5/dropout_16/dropout/GreaterEqual╚
$sequential_5/dropout_16/dropout/CastCast0sequential_5/dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_5/dropout_16/dropout/Cast█
%sequential_5/dropout_16/dropout/Mul_1Mul'sequential_5/dropout_16/dropout/Mul:z:0(sequential_5/dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_5/dropout_16/dropout/Mul_1Л
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp┘
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/dropout/Mul_1:z:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/MatMul¤
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp┌
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/BiasAddЏ
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_5/dense_16/ReluЊ
%sequential_5/dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_5/dropout_17/dropout/Constя
#sequential_5/dropout_17/dropout/MulMul(sequential_5/dense_16/Relu:activations:0.sequential_5/dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_5/dropout_17/dropout/Mulд
%sequential_5/dropout_17/dropout/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dropout_17/dropout/Shape§
<sequential_5/dropout_17/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_5/dropout_17/dropout/random_uniform/RandomUniformЦ
.sequential_5/dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_5/dropout_17/dropout/GreaterEqual/yЪ
,sequential_5/dropout_17/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_17/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_5/dropout_17/dropout/GreaterEqual╚
$sequential_5/dropout_17/dropout/CastCast0sequential_5/dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_5/dropout_17/dropout/Cast█
%sequential_5/dropout_17/dropout/Mul_1Mul'sequential_5/dropout_17/dropout/Mul:z:0(sequential_5/dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_5/dropout_17/dropout/Mul_1Е
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_17/MatMul/ReadVariableOp▒
dense_17/MatMulMatMul)sequential_5/dropout_17/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMulД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЦ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_17/SoftmaxТ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulя
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulП
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulЁ	
IdentityIdentitydense_17/Softmax:softmax:03^conv2d_15/kernel/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^sequential_5/batch_normalization_5/AssignNewValue4^sequential_5/batch_normalization_5/AssignNewValue_1C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1sequential_5/batch_normalization_5/AssignNewValue1sequential_5/batch_normalization_5/AssignNewValue2j
3sequential_5/batch_normalization_5/AssignNewValue_13sequential_5/batch_normalization_5/AssignNewValue_12ѕ
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Лџ
Ш
H__inference_sequential_5_layer_call_and_return_conditional_losses_878535
lambda_5_input;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: C
(conv2d_16_conv2d_readvariableop_resource: ђ8
)conv2d_16_biasadd_readvariableop_resource:	ђD
(conv2d_17_conv2d_readvariableop_resource:ђђ8
)conv2d_17_biasadd_readvariableop_resource:	ђ<
'dense_15_matmul_readvariableop_resource:ђбђ7
(dense_15_biasadd_readvariableop_resource:	ђ;
'dense_16_matmul_readvariableop_resource:
ђђ7
(dense_16_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpб2conv2d_15/kernel/Regularizer/Square/ReadVariableOpб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpбdense_15/BiasAdd/ReadVariableOpбdense_15/MatMul/ReadVariableOpб1dense_15/kernel/Regularizer/Square/ReadVariableOpбdense_16/BiasAdd/ReadVariableOpбdense_16/MatMul/ReadVariableOpб1dense_16/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_5/strided_slice/stackЎ
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_5/strided_slice/stack_1Ў
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_5/strided_slice/stack_2▓
lambda_5/strided_sliceStridedSlicelambda_5_input%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_5/strided_sliceХ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ж
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3lambda_5/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_5/FusedBatchNormV3░
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue╝
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1│
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOpт
conv2d_15/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_15/Conv2Dф
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp░
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_15/Relu╩
max_pooling2d_15/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool┤
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_16/Conv2D/ReadVariableOpП
conv2d_16/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_16/Conv2DФ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp▒
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/BiasAdd
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_16/Relu╦
max_pooling2d_16/MaxPoolMaxPoolconv2d_16/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPoolх
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_17/Conv2D/ReadVariableOpП
conv2d_17/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_17/Conv2DФ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp▒
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_17/Relu╦
max_pooling2d_17/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPooly
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_15/dropout/ConstИ
dropout_15/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_15/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_15/dropout/MulЁ
dropout_15/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shapeя
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype021
/dropout_15/dropout/random_uniform/RandomUniformІ
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_15/dropout/GreaterEqual/yз
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2!
dropout_15/dropout/GreaterEqualЕ
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout_15/dropout/Cast»
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout_15/dropout/Mul_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_5/ConstЮ
flatten_5/ReshapeReshapedropout_15/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_5/ReshapeФ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_15/MatMul/ReadVariableOpБ
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/MatMulе
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_15/BiasAdd/ReadVariableOpд
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_15/Reluy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Constф
dropout_16/dropout/MulMuldense_15/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shapeо
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_16/dropout/random_uniform/RandomUniformІ
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/yв
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_16/dropout/GreaterEqualА
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_16/dropout/CastД
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_16/dropout/Mul_1ф
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_16/MatMul/ReadVariableOpЦ
dense_16/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/MatMulе
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_16/BiasAdd/ReadVariableOpд
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_16/Reluy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_17/dropout/Constф
dropout_17/dropout/MulMuldense_16/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shapeо
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_17/dropout/random_uniform/RandomUniformІ
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_17/dropout/GreaterEqual/yв
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_17/dropout/GreaterEqualА
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_17/dropout/CastД
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_17/dropout/Mul_1┘
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_15/kernel/Regularizer/SquareА
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/Const┬
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/SumЇ
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_15/kernel/Regularizer/mul/x─
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mulЛ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp╣
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_15/kernel/Regularizer/SquareЌ
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/ConstЙ
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/SumІ
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_15/kernel/Regularizer/mul/x└
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulл
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOpИ
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_16/kernel/Regularizer/SquareЌ
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/ConstЙ
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/SumІ
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_16/kernel/Regularizer/mul/x└
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulш
IdentityIdentitydropout_17/dropout/Mul_1:z:0%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_5_input
┐
└
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_876198

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
 
_user_specified_nameinputs"╠L
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
StatefulPartitionedCall:0         tensorflow/serving/predict:ЗЃ
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
ж__call__
+Ж&call_and_return_all_conditional_losses
в_default_save_signature
	Вcall"ї	
_tf_keras_modelЫ{"name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
	variables
trainable_variables
regularization_losses
	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses"┘d
_tf_keras_sequential║d{"name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_5_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_5_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_5_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
О

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
№__call__
+­&call_and_return_all_conditional_losses"░
_tf_keras_layerќ{"name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
в
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m¤'mл*mЛ+mм,mМ-mн.mН/mо0mО1mп2m┘3m┌v█v▄&vП'vя*v▀+vЯ,vр-vР.vс/vС0vт1vТ2vу3vУ"
	optimizer
ќ
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
є
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
ж__call__
в_default_save_signature
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
-
ыserving_default"
signature_map
п
9	variables
:regularization_losses
;trainable_variables
<	keras_api
Ы__call__
+з&call_and_return_all_conditional_losses"К
_tf_keras_layerГ{"name": "lambda_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
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
З__call__
+ш&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
б

*kernel
+bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
Ш__call__
+э&call_and_return_all_conditional_losses"ч	
_tf_keras_layerр	{"name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
Э__call__
+щ&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
о


,kernel
-bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
Щ__call__
+ч&call_and_return_all_conditional_losses"»	
_tf_keras_layerЋ	{"name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
Ч__call__
+§&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
п


.kernel
/bias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
■__call__
+ &call_and_return_all_conditional_losses"▒	
_tf_keras_layerЌ	{"name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
ђ__call__
+Ђ&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
Ђ
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses"­
_tf_keras_layerо{"name": "dropout_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
ў
^	variables
_regularization_losses
`trainable_variables
a	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"Є
_tf_keras_layerь{"name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
ф	

0kernel
1bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
є__call__
+Є&call_and_return_all_conditional_losses"Ѓ
_tf_keras_layerж{"name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
Ђ
f	variables
gregularization_losses
htrainable_variables
i	keras_api
ѕ__call__
+Ѕ&call_and_return_all_conditional_losses"­
_tf_keras_layerо{"name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
д	

2kernel
3bias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
і__call__
+І&call_and_return_all_conditional_losses" 
_tf_keras_layerт{"name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Ђ
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses"­
_tf_keras_layerо{"name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
є
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
ј0
Ј1
љ2"
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
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
": 	ђ2dense_17/kernel
:2dense_17/bias
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
№__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
*:( 2conv2d_15/kernel
: 2conv2d_15/bias
+:) ђ2conv2d_16/kernel
:ђ2conv2d_16/bias
,:*ђђ2conv2d_17/kernel
:ђ2conv2d_17/bias
$:"ђбђ2dense_15/kernel
:ђ2dense_15/bias
#:!
ђђ2dense_16/kernel
:ђ2dense_16/bias
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
ђnon_trainable_variables
:regularization_losses
;trainable_variables
Ђlayers
 ѓlayer_regularization_losses
Ы__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
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
х
Ѓlayer_metrics
>	variables
ёmetrics
Ёnon_trainable_variables
?regularization_losses
@trainable_variables
єlayers
 Єlayer_regularization_losses
З__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
(
ј0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
х
ѕlayer_metrics
B	variables
Ѕmetrics
іnon_trainable_variables
Cregularization_losses
Dtrainable_variables
Іlayers
 їlayer_regularization_losses
Ш__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Їlayer_metrics
F	variables
јmetrics
Јnon_trainable_variables
Gregularization_losses
Htrainable_variables
љlayers
 Љlayer_regularization_losses
Э__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
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
х
њlayer_metrics
J	variables
Њmetrics
ћnon_trainable_variables
Kregularization_losses
Ltrainable_variables
Ћlayers
 ќlayer_regularization_losses
Щ__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ќlayer_metrics
N	variables
ўmetrics
Ўnon_trainable_variables
Oregularization_losses
Ptrainable_variables
џlayers
 Џlayer_regularization_losses
Ч__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
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
х
юlayer_metrics
R	variables
Юmetrics
ъnon_trainable_variables
Sregularization_losses
Ttrainable_variables
Ъlayers
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
х
Аlayer_metrics
V	variables
бmetrics
Бnon_trainable_variables
Wregularization_losses
Xtrainable_variables
цlayers
 Цlayer_regularization_losses
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
дlayer_metrics
Z	variables
Дmetrics
еnon_trainable_variables
[regularization_losses
\trainable_variables
Еlayers
 фlayer_regularization_losses
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Фlayer_metrics
^	variables
гmetrics
Гnon_trainable_variables
_regularization_losses
`trainable_variables
«layers
 »layer_regularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
(
Ј0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
х
░layer_metrics
b	variables
▒metrics
▓non_trainable_variables
cregularization_losses
dtrainable_variables
│layers
 ┤layer_regularization_losses
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
хlayer_metrics
f	variables
Хmetrics
иnon_trainable_variables
gregularization_losses
htrainable_variables
Иlayers
 ╣layer_regularization_losses
ѕ__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
(
љ0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
х
║layer_metrics
j	variables
╗metrics
╝non_trainable_variables
kregularization_losses
ltrainable_variables
йlayers
 Йlayer_regularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┐layer_metrics
n	variables
└metrics
┴non_trainable_variables
oregularization_losses
ptrainable_variables
┬layers
 ├layer_regularization_losses
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
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
п

─total

┼count
к	variables
К	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
Џ

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api"¤
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
ј0"
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
Ј0"
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
љ0"
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
к	variables"
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
':%	ђ2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
.:,2"Adam/batch_normalization_5/gamma/m
-:+2!Adam/batch_normalization_5/beta/m
/:- 2Adam/conv2d_15/kernel/m
!: 2Adam/conv2d_15/bias/m
0:. ђ2Adam/conv2d_16/kernel/m
": ђ2Adam/conv2d_16/bias/m
1:/ђђ2Adam/conv2d_17/kernel/m
": ђ2Adam/conv2d_17/bias/m
):'ђбђ2Adam/dense_15/kernel/m
!:ђ2Adam/dense_15/bias/m
(:&
ђђ2Adam/dense_16/kernel/m
!:ђ2Adam/dense_16/bias/m
':%	ђ2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
.:,2"Adam/batch_normalization_5/gamma/v
-:+2!Adam/batch_normalization_5/beta/v
/:- 2Adam/conv2d_15/kernel/v
!: 2Adam/conv2d_15/bias/v
0:. ђ2Adam/conv2d_16/kernel/v
": ђ2Adam/conv2d_16/bias/v
1:/ђђ2Adam/conv2d_17/kernel/v
": ђ2Adam/conv2d_17/bias/v
):'ђбђ2Adam/dense_15/kernel/v
!:ђ2Adam/dense_15/bias/v
(:&
ђђ2Adam/dense_16/kernel/v
!:ђ2Adam/dense_16/bias/v
м2¤
$__inference_CNN_layer_call_fn_877498
$__inference_CNN_layer_call_fn_877535
$__inference_CNN_layer_call_fn_877572
$__inference_CNN_layer_call_fn_877609┤
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
Й2╗
?__inference_CNN_layer_call_and_return_conditional_losses_877699
?__inference_CNN_layer_call_and_return_conditional_losses_877810
?__inference_CNN_layer_call_and_return_conditional_losses_877900
?__inference_CNN_layer_call_and_return_conditional_losses_878011┤
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
!__inference__wrapped_model_876132Й
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
__inference_call_799973
__inference_call_800045
__inference_call_800117│
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
-__inference_sequential_5_layer_call_fn_878062
-__inference_sequential_5_layer_call_fn_878095
-__inference_sequential_5_layer_call_fn_878128
-__inference_sequential_5_layer_call_fn_878161└
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
H__inference_sequential_5_layer_call_and_return_conditional_losses_878244
H__inference_sequential_5_layer_call_and_return_conditional_losses_878348
H__inference_sequential_5_layer_call_and_return_conditional_losses_878431
H__inference_sequential_5_layer_call_and_return_conditional_losses_878535└
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
)__inference_dense_17_layer_call_fn_878544б
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
D__inference_dense_17_layer_call_and_return_conditional_losses_878555б
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
$__inference_signature_wrapper_877461input_1"ћ
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
)__inference_lambda_5_layer_call_fn_878560
)__inference_lambda_5_layer_call_fn_878565└
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
D__inference_lambda_5_layer_call_and_return_conditional_losses_878573
D__inference_lambda_5_layer_call_and_return_conditional_losses_878581└
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
6__inference_batch_normalization_5_layer_call_fn_878594
6__inference_batch_normalization_5_layer_call_fn_878607
6__inference_batch_normalization_5_layer_call_fn_878620
6__inference_batch_normalization_5_layer_call_fn_878633┤
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878651
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878669
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878687
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878705┤
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
*__inference_conv2d_15_layer_call_fn_878720б
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
E__inference_conv2d_15_layer_call_and_return_conditional_losses_878737б
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
1__inference_max_pooling2d_15_layer_call_fn_876270Я
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
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_876264Я
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
*__inference_conv2d_16_layer_call_fn_878746б
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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_878757б
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
1__inference_max_pooling2d_16_layer_call_fn_876282Я
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
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_876276Я
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
*__inference_conv2d_17_layer_call_fn_878766б
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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_878777б
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
1__inference_max_pooling2d_17_layer_call_fn_876294Я
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
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_876288Я
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
+__inference_dropout_15_layer_call_fn_878782
+__inference_dropout_15_layer_call_fn_878787┤
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
F__inference_dropout_15_layer_call_and_return_conditional_losses_878792
F__inference_dropout_15_layer_call_and_return_conditional_losses_878804┤
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
*__inference_flatten_5_layer_call_fn_878809б
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_878815б
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
)__inference_dense_15_layer_call_fn_878830б
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
D__inference_dense_15_layer_call_and_return_conditional_losses_878847б
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
+__inference_dropout_16_layer_call_fn_878852
+__inference_dropout_16_layer_call_fn_878857┤
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_878862
F__inference_dropout_16_layer_call_and_return_conditional_losses_878874┤
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
)__inference_dense_16_layer_call_fn_878889б
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
D__inference_dense_16_layer_call_and_return_conditional_losses_878906б
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
+__inference_dropout_17_layer_call_fn_878911
+__inference_dropout_17_layer_call_fn_878916┤
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_878921
F__inference_dropout_17_layer_call_and_return_conditional_losses_878933┤
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
__inference_loss_fn_0_878944Ј
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
__inference_loss_fn_1_878955Ј
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
__inference_loss_fn_2_878966Ј
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
annotationsф *б ╣
?__inference_CNN_layer_call_and_return_conditional_losses_877699v&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p 
ф "%б"
і
0         
џ ╣
?__inference_CNN_layer_call_and_return_conditional_losses_877810v&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p
ф "%б"
і
0         
џ ║
?__inference_CNN_layer_call_and_return_conditional_losses_877900w&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p 
ф "%б"
і
0         
џ ║
?__inference_CNN_layer_call_and_return_conditional_losses_878011w&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p
ф "%б"
і
0         
џ њ
$__inference_CNN_layer_call_fn_877498j&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p 
ф "і         Љ
$__inference_CNN_layer_call_fn_877535i&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p 
ф "і         Љ
$__inference_CNN_layer_call_fn_877572i&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p
ф "і         њ
$__inference_CNN_layer_call_fn_877609j&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p
ф "і         Д
!__inference__wrapped_model_876132Ђ&'()*+,-./01238б5
.б+
)і&
input_1         KK
ф "3ф0
.
output_1"і
output_1         В
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878651ќ&'()MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ В
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878669ќ&'()MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ К
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878687r&'();б8
1б.
(і%
inputs         KK
p 
ф "-б*
#і 
0         KK
џ К
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_878705r&'();б8
1б.
(і%
inputs         KK
p
ф "-б*
#і 
0         KK
џ ─
6__inference_batch_normalization_5_layer_call_fn_878594Ѕ&'()MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ─
6__inference_batch_normalization_5_layer_call_fn_878607Ѕ&'()MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Ъ
6__inference_batch_normalization_5_layer_call_fn_878620e&'();б8
1б.
(і%
inputs         KK
p 
ф " і         KKЪ
6__inference_batch_normalization_5_layer_call_fn_878633e&'();б8
1б.
(і%
inputs         KK
p
ф " і         KKt
__inference_call_799973Y&'()*+,-./01233б0
)б&
 і
inputsђKK
p
ф "і	ђt
__inference_call_800045Y&'()*+,-./01233б0
)б&
 і
inputsђKK
p 
ф "і	ђё
__inference_call_800117i&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p 
ф "і         х
E__inference_conv2d_15_layer_call_and_return_conditional_losses_878737l*+7б4
-б*
(і%
inputs         KK
ф "-б*
#і 
0         KK 
џ Ї
*__inference_conv2d_15_layer_call_fn_878720_*+7б4
-б*
(і%
inputs         KK
ф " і         KK Х
E__inference_conv2d_16_layer_call_and_return_conditional_losses_878757m,-7б4
-б*
(і%
inputs         %% 
ф ".б+
$і!
0         %%ђ
џ ј
*__inference_conv2d_16_layer_call_fn_878746`,-7б4
-б*
(і%
inputs         %% 
ф "!і         %%ђи
E__inference_conv2d_17_layer_call_and_return_conditional_losses_878777n./8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ј
*__inference_conv2d_17_layer_call_fn_878766a./8б5
.б+
)і&
inputs         ђ
ф "!і         ђД
D__inference_dense_15_layer_call_and_return_conditional_losses_878847_011б.
'б$
"і
inputs         ђб
ф "&б#
і
0         ђ
џ 
)__inference_dense_15_layer_call_fn_878830R011б.
'б$
"і
inputs         ђб
ф "і         ђд
D__inference_dense_16_layer_call_and_return_conditional_losses_878906^230б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
)__inference_dense_16_layer_call_fn_878889Q230б-
&б#
!і
inputs         ђ
ф "і         ђЦ
D__inference_dense_17_layer_call_and_return_conditional_losses_878555]0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ }
)__inference_dense_17_layer_call_fn_878544P0б-
&б#
!і
inputs         ђ
ф "і         И
F__inference_dropout_15_layer_call_and_return_conditional_losses_878792n<б9
2б/
)і&
inputs         		ђ
p 
ф ".б+
$і!
0         		ђ
џ И
F__inference_dropout_15_layer_call_and_return_conditional_losses_878804n<б9
2б/
)і&
inputs         		ђ
p
ф ".б+
$і!
0         		ђ
џ љ
+__inference_dropout_15_layer_call_fn_878782a<б9
2б/
)і&
inputs         		ђ
p 
ф "!і         		ђљ
+__inference_dropout_15_layer_call_fn_878787a<б9
2б/
)і&
inputs         		ђ
p
ф "!і         		ђе
F__inference_dropout_16_layer_call_and_return_conditional_losses_878862^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_16_layer_call_and_return_conditional_losses_878874^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_16_layer_call_fn_878852Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_16_layer_call_fn_878857Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђе
F__inference_dropout_17_layer_call_and_return_conditional_losses_878921^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_17_layer_call_and_return_conditional_losses_878933^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_17_layer_call_fn_878911Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_17_layer_call_fn_878916Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђг
E__inference_flatten_5_layer_call_and_return_conditional_losses_878815c8б5
.б+
)і&
inputs         		ђ
ф "'б$
і
0         ђб
џ ё
*__inference_flatten_5_layer_call_fn_878809V8б5
.б+
)і&
inputs         		ђ
ф "і         ђбИ
D__inference_lambda_5_layer_call_and_return_conditional_losses_878573p?б<
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
D__inference_lambda_5_layer_call_and_return_conditional_losses_878581p?б<
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
)__inference_lambda_5_layer_call_fn_878560c?б<
5б2
(і%
inputs         KK

 
p 
ф " і         KKљ
)__inference_lambda_5_layer_call_fn_878565c?б<
5б2
(і%
inputs         KK

 
p
ф " і         KK;
__inference_loss_fn_0_878944*б

б 
ф "і ;
__inference_loss_fn_1_8789550б

б 
ф "і ;
__inference_loss_fn_2_8789662б

б 
ф "і №
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_876264ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_15_layer_call_fn_876270ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_876276ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_16_layer_call_fn_876282ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_876288ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_17_layer_call_fn_876294ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ┼
H__inference_sequential_5_layer_call_and_return_conditional_losses_878244y&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p 

 
ф "&б#
і
0         ђ
џ ┼
H__inference_sequential_5_layer_call_and_return_conditional_losses_878348y&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p

 
ф "&б#
і
0         ђ
џ ╬
H__inference_sequential_5_layer_call_and_return_conditional_losses_878431Ђ&'()*+,-./0123GбD
=б:
0і-
lambda_5_input         KK
p 

 
ф "&б#
і
0         ђ
џ ╬
H__inference_sequential_5_layer_call_and_return_conditional_losses_878535Ђ&'()*+,-./0123GбD
=б:
0і-
lambda_5_input         KK
p

 
ф "&б#
і
0         ђ
џ Ц
-__inference_sequential_5_layer_call_fn_878062t&'()*+,-./0123GбD
=б:
0і-
lambda_5_input         KK
p 

 
ф "і         ђЮ
-__inference_sequential_5_layer_call_fn_878095l&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p 

 
ф "і         ђЮ
-__inference_sequential_5_layer_call_fn_878128l&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p

 
ф "і         ђЦ
-__inference_sequential_5_layer_call_fn_878161t&'()*+,-./0123GбD
=б:
0і-
lambda_5_input         KK
p

 
ф "і         ђх
$__inference_signature_wrapper_877461ї&'()*+,-./0123Cб@
б 
9ф6
4
input_1)і&
input_1         KK"3ф0
.
output_1"і
output_1         