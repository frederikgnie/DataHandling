Ьи:
Т
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
ж
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
Р
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28у4

conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0

conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
dtype0

conv3d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_10/kernel

$conv3d_10/kernel/Read/ReadVariableOpReadVariableOpconv3d_10/kernel**
_output_shapes
:*
dtype0
t
conv3d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_10/bias
m
"conv3d_10/bias/Read/ReadVariableOpReadVariableOpconv3d_10/bias*
_output_shapes
:*
dtype0

conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:*
dtype0

conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
:*
dtype0
r
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
:*
dtype0

conv3d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_11/kernel

$conv3d_11/kernel/Read/ReadVariableOpReadVariableOpconv3d_11/kernel**
_output_shapes
:*
dtype0
t
conv3d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_11/bias
m
"conv3d_11/bias/Read/ReadVariableOpReadVariableOpconv3d_11/bias*
_output_shapes
:*
dtype0

conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:*
dtype0

conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_7/kernel

#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel**
_output_shapes
:*
dtype0
r
conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_7/bias
k
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes
:*
dtype0

conv3d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_12/kernel

$conv3d_12/kernel/Read/ReadVariableOpReadVariableOpconv3d_12/kernel**
_output_shapes
:*
dtype0
t
conv3d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_12/bias
m
"conv3d_12/bias/Read/ReadVariableOpReadVariableOpconv3d_12/bias*
_output_shapes
:*
dtype0

conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:*
dtype0

conv3d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_8/kernel

#conv3d_8/kernel/Read/ReadVariableOpReadVariableOpconv3d_8/kernel**
_output_shapes
:*
dtype0
r
conv3d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_8/bias
k
!conv3d_8/bias/Read/ReadVariableOpReadVariableOpconv3d_8/bias*
_output_shapes
:*
dtype0

conv3d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_13/kernel

$conv3d_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_13/kernel**
_output_shapes
:*
dtype0
t
conv3d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_13/bias
m
"conv3d_13/bias/Read/ReadVariableOpReadVariableOpconv3d_13/bias*
_output_shapes
:*
dtype0

conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0

conv3d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_9/kernel

#conv3d_9/kernel/Read/ReadVariableOpReadVariableOpconv3d_9/kernel**
_output_shapes
:*
dtype0
r
conv3d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_9/bias
k
!conv3d_9/bias/Read/ReadVariableOpReadVariableOpconv3d_9/bias*
_output_shapes
:*
dtype0

conv3d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_14/kernel

$conv3d_14/kernel/Read/ReadVariableOpReadVariableOpconv3d_14/kernel**
_output_shapes
:*
dtype0
t
conv3d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_14/bias
m
"conv3d_14/bias/Read/ReadVariableOpReadVariableOpconv3d_14/bias*
_output_shapes
:*
dtype0

conv3d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_15/kernel

$conv3d_15/kernel/Read/ReadVariableOpReadVariableOpconv3d_15/kernel**
_output_shapes
:*
dtype0
t
conv3d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_15/bias
m
"conv3d_15/bias/Read/ReadVariableOpReadVariableOpconv3d_15/bias*
_output_shapes
:*
dtype0

conv3d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_20/kernel

$conv3d_20/kernel/Read/ReadVariableOpReadVariableOpconv3d_20/kernel**
_output_shapes
:*
dtype0
t
conv3d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_20/bias
m
"conv3d_20/bias/Read/ReadVariableOpReadVariableOpconv3d_20/bias*
_output_shapes
:*
dtype0

conv3d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_25/kernel

$conv3d_25/kernel/Read/ReadVariableOpReadVariableOpconv3d_25/kernel**
_output_shapes
:*
dtype0
t
conv3d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_25/bias
m
"conv3d_25/bias/Read/ReadVariableOpReadVariableOpconv3d_25/bias*
_output_shapes
:*
dtype0

conv3d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_16/kernel

$conv3d_16/kernel/Read/ReadVariableOpReadVariableOpconv3d_16/kernel**
_output_shapes
:*
dtype0
t
conv3d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_16/bias
m
"conv3d_16/bias/Read/ReadVariableOpReadVariableOpconv3d_16/bias*
_output_shapes
:*
dtype0

conv3d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_21/kernel

$conv3d_21/kernel/Read/ReadVariableOpReadVariableOpconv3d_21/kernel**
_output_shapes
:*
dtype0
t
conv3d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_21/bias
m
"conv3d_21/bias/Read/ReadVariableOpReadVariableOpconv3d_21/bias*
_output_shapes
:*
dtype0

conv3d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_26/kernel

$conv3d_26/kernel/Read/ReadVariableOpReadVariableOpconv3d_26/kernel**
_output_shapes
:*
dtype0
t
conv3d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_26/bias
m
"conv3d_26/bias/Read/ReadVariableOpReadVariableOpconv3d_26/bias*
_output_shapes
:*
dtype0

conv3d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_17/kernel

$conv3d_17/kernel/Read/ReadVariableOpReadVariableOpconv3d_17/kernel**
_output_shapes
:*
dtype0
t
conv3d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_17/bias
m
"conv3d_17/bias/Read/ReadVariableOpReadVariableOpconv3d_17/bias*
_output_shapes
:*
dtype0

conv3d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_22/kernel

$conv3d_22/kernel/Read/ReadVariableOpReadVariableOpconv3d_22/kernel**
_output_shapes
:*
dtype0
t
conv3d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_22/bias
m
"conv3d_22/bias/Read/ReadVariableOpReadVariableOpconv3d_22/bias*
_output_shapes
:*
dtype0

conv3d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_27/kernel

$conv3d_27/kernel/Read/ReadVariableOpReadVariableOpconv3d_27/kernel**
_output_shapes
:*
dtype0
t
conv3d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_27/bias
m
"conv3d_27/bias/Read/ReadVariableOpReadVariableOpconv3d_27/bias*
_output_shapes
:*
dtype0

conv3d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_18/kernel

$conv3d_18/kernel/Read/ReadVariableOpReadVariableOpconv3d_18/kernel**
_output_shapes
:*
dtype0
t
conv3d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_18/bias
m
"conv3d_18/bias/Read/ReadVariableOpReadVariableOpconv3d_18/bias*
_output_shapes
:*
dtype0

conv3d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_23/kernel

$conv3d_23/kernel/Read/ReadVariableOpReadVariableOpconv3d_23/kernel**
_output_shapes
:*
dtype0
t
conv3d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_23/bias
m
"conv3d_23/bias/Read/ReadVariableOpReadVariableOpconv3d_23/bias*
_output_shapes
:*
dtype0

conv3d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_28/kernel

$conv3d_28/kernel/Read/ReadVariableOpReadVariableOpconv3d_28/kernel**
_output_shapes
:*
dtype0
t
conv3d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_28/bias
m
"conv3d_28/bias/Read/ReadVariableOpReadVariableOpconv3d_28/bias*
_output_shapes
:*
dtype0

conv3d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_19/kernel

$conv3d_19/kernel/Read/ReadVariableOpReadVariableOpconv3d_19/kernel**
_output_shapes
:*
dtype0
t
conv3d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_19/bias
m
"conv3d_19/bias/Read/ReadVariableOpReadVariableOpconv3d_19/bias*
_output_shapes
:*
dtype0

conv3d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_24/kernel

$conv3d_24/kernel/Read/ReadVariableOpReadVariableOpconv3d_24/kernel**
_output_shapes
:*
dtype0
t
conv3d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_24/bias
m
"conv3d_24/bias/Read/ReadVariableOpReadVariableOpconv3d_24/bias*
_output_shapes
:*
dtype0

conv3d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_29/kernel

$conv3d_29/kernel/Read/ReadVariableOpReadVariableOpconv3d_29/kernel**
_output_shapes
:*
dtype0
t
conv3d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_29/bias
m
"conv3d_29/bias/Read/ReadVariableOpReadVariableOpconv3d_29/bias*
_output_shapes
:*
dtype0

conv3d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_30/kernel

$conv3d_30/kernel/Read/ReadVariableOpReadVariableOpconv3d_30/kernel**
_output_shapes
:*
dtype0
t
conv3d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_30/bias
m
"conv3d_30/bias/Read/ReadVariableOpReadVariableOpconv3d_30/bias*
_output_shapes
:*
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

Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/m

(Adam/conv3d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/m**
_output_shapes
:*
dtype0
|
Adam/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/m
u
&Adam/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/m

*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/m
y
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_10/kernel/m

+Adam/conv3d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_10/bias/m
{
)Adam/conv3d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_1/kernel/m

*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_1/bias/m
y
(Adam/conv3d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/m

*Adam/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_6/bias/m
y
(Adam/conv3d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_11/kernel/m

+Adam/conv3d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_11/bias/m
{
)Adam/conv3d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_2/kernel/m

*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_2/bias/m
y
(Adam/conv3d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/m

*Adam/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_7/bias/m
y
(Adam/conv3d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_12/kernel/m

+Adam/conv3d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_12/bias/m
{
)Adam/conv3d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/m

*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/m
y
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_8/kernel/m

*Adam/conv3d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_8/bias/m
y
(Adam/conv3d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_13/kernel/m

+Adam/conv3d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_13/bias/m
{
)Adam/conv3d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_4/kernel/m

*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/m
y
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_9/kernel/m

*Adam/conv3d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_9/bias/m
y
(Adam/conv3d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_14/kernel/m

+Adam/conv3d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_14/bias/m
{
)Adam/conv3d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_15/kernel/m

+Adam/conv3d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_15/bias/m
{
)Adam/conv3d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_20/kernel/m

+Adam/conv3d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_20/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_20/bias/m
{
)Adam/conv3d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_20/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_25/kernel/m

+Adam/conv3d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_25/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_25/bias/m
{
)Adam/conv3d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_25/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_16/kernel/m

+Adam/conv3d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_16/bias/m
{
)Adam/conv3d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_21/kernel/m

+Adam/conv3d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_21/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_21/bias/m
{
)Adam/conv3d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_21/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_26/kernel/m

+Adam/conv3d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_26/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_26/bias/m
{
)Adam/conv3d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_26/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_17/kernel/m

+Adam/conv3d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_17/bias/m
{
)Adam/conv3d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_22/kernel/m

+Adam/conv3d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_22/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_22/bias/m
{
)Adam/conv3d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_22/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_27/kernel/m

+Adam/conv3d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_27/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_27/bias/m
{
)Adam/conv3d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_27/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_18/kernel/m

+Adam/conv3d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_18/bias/m
{
)Adam/conv3d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_23/kernel/m

+Adam/conv3d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_23/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_23/bias/m
{
)Adam/conv3d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_23/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_28/kernel/m

+Adam/conv3d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_28/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_28/bias/m
{
)Adam/conv3d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_28/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_19/kernel/m

+Adam/conv3d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_19/bias/m
{
)Adam/conv3d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_24/kernel/m

+Adam/conv3d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_24/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_24/bias/m
{
)Adam/conv3d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_24/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_29/kernel/m

+Adam/conv3d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_29/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_29/bias/m
{
)Adam/conv3d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_29/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_30/kernel/m

+Adam/conv3d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_30/kernel/m**
_output_shapes
:*
dtype0

Adam/conv3d_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_30/bias/m
{
)Adam/conv3d_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_30/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/v

(Adam/conv3d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/v**
_output_shapes
:*
dtype0
|
Adam/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/v
u
&Adam/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/v

*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/v
y
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_10/kernel/v

+Adam/conv3d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_10/bias/v
{
)Adam/conv3d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_1/kernel/v

*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_1/bias/v
y
(Adam/conv3d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/v

*Adam/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_6/bias/v
y
(Adam/conv3d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_11/kernel/v

+Adam/conv3d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_11/bias/v
{
)Adam/conv3d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_2/kernel/v

*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_2/bias/v
y
(Adam/conv3d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/v

*Adam/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_7/bias/v
y
(Adam/conv3d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_12/kernel/v

+Adam/conv3d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_12/bias/v
{
)Adam/conv3d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/v

*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/v
y
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_8/kernel/v

*Adam/conv3d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_8/bias/v
y
(Adam/conv3d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_13/kernel/v

+Adam/conv3d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_13/bias/v
{
)Adam/conv3d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_4/kernel/v

*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/v
y
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_9/kernel/v

*Adam/conv3d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_9/bias/v
y
(Adam/conv3d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_14/kernel/v

+Adam/conv3d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_14/bias/v
{
)Adam/conv3d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_15/kernel/v

+Adam/conv3d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_15/bias/v
{
)Adam/conv3d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_20/kernel/v

+Adam/conv3d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_20/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_20/bias/v
{
)Adam/conv3d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_20/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_25/kernel/v

+Adam/conv3d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_25/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_25/bias/v
{
)Adam/conv3d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_25/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_16/kernel/v

+Adam/conv3d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_16/bias/v
{
)Adam/conv3d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_21/kernel/v

+Adam/conv3d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_21/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_21/bias/v
{
)Adam/conv3d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_21/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_26/kernel/v

+Adam/conv3d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_26/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_26/bias/v
{
)Adam/conv3d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_26/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_17/kernel/v

+Adam/conv3d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_17/bias/v
{
)Adam/conv3d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_22/kernel/v

+Adam/conv3d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_22/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_22/bias/v
{
)Adam/conv3d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_22/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_27/kernel/v

+Adam/conv3d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_27/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_27/bias/v
{
)Adam/conv3d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_27/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_18/kernel/v

+Adam/conv3d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_18/bias/v
{
)Adam/conv3d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_23/kernel/v

+Adam/conv3d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_23/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_23/bias/v
{
)Adam/conv3d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_23/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_28/kernel/v

+Adam/conv3d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_28/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_28/bias/v
{
)Adam/conv3d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_28/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_19/kernel/v

+Adam/conv3d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_19/bias/v
{
)Adam/conv3d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_24/kernel/v

+Adam/conv3d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_24/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_24/bias/v
{
)Adam/conv3d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_24/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_29/kernel/v

+Adam/conv3d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_29/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_29/bias/v
{
)Adam/conv3d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_29/bias/v*
_output_shapes
:*
dtype0

Adam/conv3d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_30/kernel/v

+Adam/conv3d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_30/kernel/v**
_output_shapes
:*
dtype0

Adam/conv3d_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_30/bias/v
{
)Adam/conv3d_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_30/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ѕв
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Џв
valueЄвB в Bв

layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-15
 layer-31
!layer_with_weights-16
!layer-32
"layer_with_weights-17
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-18
&layer-37
'layer_with_weights-19
'layer-38
(layer_with_weights-20
(layer-39
)layer_with_weights-21
)layer-40
*layer_with_weights-22
*layer-41
+layer_with_weights-23
+layer-42
,layer-43
-layer-44
.layer-45
/layer_with_weights-24
/layer-46
0layer_with_weights-25
0layer-47
1layer_with_weights-26
1layer-48
2layer_with_weights-27
2layer-49
3layer_with_weights-28
3layer-50
4layer_with_weights-29
4layer-51
5layer-52
6layer_with_weights-30
6layer-53
7	optimizer
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<
signatures
 
 
 
R
=	variables
>trainable_variables
?regularization_losses
@	keras_api
R
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
h

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
h

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
h

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
h

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
R
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
R
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
R
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
k

}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
 	keras_api
V
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
V
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
V
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
n
­kernel
	Ўbias
Џ	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
n
Гkernel
	Дbias
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
n
Йkernel
	Кbias
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
V
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
V
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
V
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
n
Ыkernel
	Ьbias
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
n
бkernel
	вbias
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
n
зkernel
	иbias
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
V
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
V
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
V
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
n
щkernel
	ъbias
ы	variables
ьtrainable_variables
эregularization_losses
ю	keras_api
n
яkernel
	№bias
ё	variables
ђtrainable_variables
ѓregularization_losses
є	keras_api
n
ѕkernel
	іbias
ї	variables
јtrainable_variables
љregularization_losses
њ	keras_api
n
ћkernel
	ќbias
§	variables
ўtrainable_variables
џregularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	 bias
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
n
Ѕkernel
	Іbias
Ї	variables
Јtrainable_variables
Љregularization_losses
Њ	keras_api
n
Ћkernel
	Ќbias
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
n
Бkernel
	Вbias
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
n
Зkernel
	Иbias
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
V
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
n
Сkernel
	Тbias
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
§

	Чiter
Шbeta_1
Щbeta_2

Ъdecay
Ыlearning_rateMmеNmжSmзTmиYmйZmк_mл`mмemнfmоkmпlmр}mс~mт	mу	mф	mх	mц	mч	mш	mщ	mъ	mы	mь	­mэ	Ўmю	Гmя	Дm№	Йmё	Кmђ	Ыmѓ	Ьmє	бmѕ	вmі	зmї	иmј	щmљ	ъmњ	яmћ	№mќ	ѕm§	іmў	ћmџ	ќm	m	m	m	m	m	m	m	 m	Ѕm	Іm	Ћm	Ќm	Бm	Вm	Зm	Иm	Сm	ТmMvNvSvTvYvZv_v`vevfvkvlv}v~v 	vЁ	vЂ	vЃ	vЄ	vЅ	vІ	vЇ	vЈ	vЉ	vЊ	­vЋ	ЎvЌ	Гv­	ДvЎ	ЙvЏ	КvА	ЫvБ	ЬvВ	бvГ	вvД	зvЕ	иvЖ	щvЗ	ъvИ	яvЙ	№vК	ѕvЛ	іvМ	ћvН	ќvО	vП	vР	vС	vТ	vУ	vФ	vХ	 vЦ	ЅvЧ	ІvШ	ЋvЩ	ЌvЪ	БvЫ	ВvЬ	ЗvЭ	ИvЮ	СvЯ	Тvа

M0
N1
S2
T3
Y4
Z5
_6
`7
e8
f9
k10
l11
}12
~13
14
15
16
17
18
19
20
21
22
23
­24
Ў25
Г26
Д27
Й28
К29
Ы30
Ь31
б32
в33
з34
и35
щ36
ъ37
я38
№39
ѕ40
і41
ћ42
ќ43
44
45
46
47
48
49
50
 51
Ѕ52
І53
Ћ54
Ќ55
Б56
В57
З58
И59
С60
Т61

M0
N1
S2
T3
Y4
Z5
_6
`7
e8
f9
k10
l11
}12
~13
14
15
16
17
18
19
20
21
22
23
­24
Ў25
Г26
Д27
Й28
К29
Ы30
Ь31
б32
в33
з34
и35
щ36
ъ37
я38
№39
ѕ40
і41
ћ42
ќ43
44
45
46
47
48
49
50
 51
Ѕ52
І53
Ћ54
Ќ55
Б56
В57
З58
И59
С60
Т61
 
В
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 
 
 
В
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
=	variables
>trainable_variables
?regularization_losses
 
 
 
В
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
 
 
В
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
В
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

M0
N1
 
В
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
В
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
\Z
VARIABLE_VALUEconv3d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
В
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
[	variables
\trainable_variables
]regularization_losses
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
 
В
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
a	variables
btrainable_variables
cregularization_losses
[Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
В
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
g	variables
htrainable_variables
iregularization_losses
\Z
VARIABLE_VALUEconv3d_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

k0
l1
 
В
ўnon_trainable_variables
џlayers
metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
 
 
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
q	variables
rtrainable_variables
sregularization_losses
 
 
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
 
 
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
y	variables
ztrainable_variables
{regularization_losses
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1

}0
~1
 
Д
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEconv3d_12/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_12/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEconv3d_8/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_8/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEconv3d_13/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_13/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
 
 
 
Е
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
 
 
 
Е
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
\Z
VARIABLE_VALUEconv3d_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_4/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

­0
Ў1

­0
Ў1
 
Е
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
Џ	variables
Аtrainable_variables
Бregularization_losses
\Z
VARIABLE_VALUEconv3d_9/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_9/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

Г0
Д1

Г0
Д1
 
Е
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
][
VARIABLE_VALUEconv3d_14/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_14/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

Й0
К1

Й0
К1
 
Е
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
 
 
 
Е
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
 
 
 
Е
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
 
 
 
Е
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
][
VARIABLE_VALUEconv3d_15/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_15/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

Ы0
Ь1

Ы0
Ь1
 
Е
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
][
VARIABLE_VALUEconv3d_20/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_20/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

б0
в1

б0
в1
 
Е
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
г	variables
дtrainable_variables
еregularization_losses
][
VARIABLE_VALUEconv3d_25/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_25/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

з0
и1

з0
и1
 
Е
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
 
 
 
Е
ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
н	variables
оtrainable_variables
пregularization_losses
 
 
 
Е
ёnon_trainable_variables
ђlayers
ѓmetrics
 єlayer_regularization_losses
ѕlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
 
 
 
Е
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
][
VARIABLE_VALUEconv3d_16/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_16/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

щ0
ъ1

щ0
ъ1
 
Е
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
ы	variables
ьtrainable_variables
эregularization_losses
][
VARIABLE_VALUEconv3d_21/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_21/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

я0
№1

я0
№1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ё	variables
ђtrainable_variables
ѓregularization_losses
][
VARIABLE_VALUEconv3d_26/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_26/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

ѕ0
і1

ѕ0
і1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ї	variables
јtrainable_variables
љregularization_losses
][
VARIABLE_VALUEconv3d_17/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_17/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

ћ0
ќ1

ћ0
ќ1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
ўtrainable_variables
џregularization_losses
][
VARIABLE_VALUEconv3d_22/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_22/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEconv3d_27/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_27/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
Ѓnon_trainable_variables
Єlayers
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEconv3d_18/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_18/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEconv3d_23/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_23/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
Е
­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
][
VARIABLE_VALUEconv3d_28/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_28/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE

Ѕ0
І1

Ѕ0
І1
 
Е
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Ї	variables
Јtrainable_variables
Љregularization_losses
][
VARIABLE_VALUEconv3d_19/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_19/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE

Ћ0
Ќ1

Ћ0
Ќ1
 
Е
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
][
VARIABLE_VALUEconv3d_24/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_24/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE

Б0
В1

Б0
В1
 
Е
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
][
VARIABLE_VALUEconv3d_29/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_29/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE

З0
И1

З0
И1
 
Е
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
 
 
 
Е
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
][
VARIABLE_VALUEconv3d_30/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_30/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE

С0
Т1

С0
Т1
 
Е
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
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
 
І
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653

а0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

бtotal

вcount
г	variables
д	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

б0
в1

г	variables
|z
VARIABLE_VALUEAdam/conv3d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_11/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_11/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_12/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_12/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_8/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_8/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_13/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_13/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_4/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_4/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_9/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_9/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_14/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_14/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_15/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_15/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_20/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_20/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_25/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_25/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_16/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_16/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_21/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_21/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_26/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_26/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_17/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_17/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_22/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_22/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_27/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_27/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_18/kernel/mSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_18/bias/mQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_23/kernel/mSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_23/bias/mQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_28/kernel/mSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_28/bias/mQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_19/kernel/mSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_19/bias/mQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_24/kernel/mSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_24/bias/mQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_29/kernel/mSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_29/bias/mQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_30/kernel/mSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_30/bias/mQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_11/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_11/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_12/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_12/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_8/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_8/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_13/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_13/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_4/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_4/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_9/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_9/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_14/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_14/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_15/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_15/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_20/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_20/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_25/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_25/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_16/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_16/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_21/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_21/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_26/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_26/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_17/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_17/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_22/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_22/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_27/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_27/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_18/kernel/vSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_18/bias/vQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_23/kernel/vSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_23/bias/vQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_28/kernel/vSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_28/bias/vQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_19/kernel/vSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_19/bias/vQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_24/kernel/vSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_24/bias/vQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_29/kernel/vSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_29/bias/vQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv3d_30/kernel/vSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_30/bias/vQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_u_velPlaceholder*/
_output_shapes
:џџџџџџџџџ   *
dtype0*$
shape:џџџџџџџџџ   

serving_default_v_velPlaceholder*/
_output_shapes
:џџџџџџџџџ   *
dtype0*$
shape:џџџџџџџџџ   

serving_default_w_velPlaceholder*/
_output_shapes
:џџџџџџџџџ   *
dtype0*$
shape:џџџџџџџџџ   
ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_u_velserving_default_v_velserving_default_w_velconv3d_10/kernelconv3d_10/biasconv3d_5/kernelconv3d_5/biasconv3d/kernelconv3d/biasconv3d_11/kernelconv3d_11/biasconv3d_6/kernelconv3d_6/biasconv3d_1/kernelconv3d_1/biasconv3d_12/kernelconv3d_12/biasconv3d_7/kernelconv3d_7/biasconv3d_2/kernelconv3d_2/biasconv3d_13/kernelconv3d_13/biasconv3d_8/kernelconv3d_8/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_9/kernelconv3d_9/biasconv3d_14/kernelconv3d_14/biasconv3d_25/kernelconv3d_25/biasconv3d_20/kernelconv3d_20/biasconv3d_15/kernelconv3d_15/biasconv3d_26/kernelconv3d_26/biasconv3d_21/kernelconv3d_21/biasconv3d_16/kernelconv3d_16/biasconv3d_27/kernelconv3d_27/biasconv3d_22/kernelconv3d_22/biasconv3d_17/kernelconv3d_17/biasconv3d_28/kernelconv3d_28/biasconv3d_23/kernelconv3d_23/biasconv3d_18/kernelconv3d_18/biasconv3d_19/kernelconv3d_19/biasconv3d_24/kernelconv3d_24/biasconv3d_29/kernelconv3d_29/biasconv3d_30/kernelconv3d_30/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_134152
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ТA
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp$conv3d_10/kernel/Read/ReadVariableOp"conv3d_10/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp$conv3d_11/kernel/Read/ReadVariableOp"conv3d_11/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp$conv3d_12/kernel/Read/ReadVariableOp"conv3d_12/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_8/kernel/Read/ReadVariableOp!conv3d_8/bias/Read/ReadVariableOp$conv3d_13/kernel/Read/ReadVariableOp"conv3d_13/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_9/kernel/Read/ReadVariableOp!conv3d_9/bias/Read/ReadVariableOp$conv3d_14/kernel/Read/ReadVariableOp"conv3d_14/bias/Read/ReadVariableOp$conv3d_15/kernel/Read/ReadVariableOp"conv3d_15/bias/Read/ReadVariableOp$conv3d_20/kernel/Read/ReadVariableOp"conv3d_20/bias/Read/ReadVariableOp$conv3d_25/kernel/Read/ReadVariableOp"conv3d_25/bias/Read/ReadVariableOp$conv3d_16/kernel/Read/ReadVariableOp"conv3d_16/bias/Read/ReadVariableOp$conv3d_21/kernel/Read/ReadVariableOp"conv3d_21/bias/Read/ReadVariableOp$conv3d_26/kernel/Read/ReadVariableOp"conv3d_26/bias/Read/ReadVariableOp$conv3d_17/kernel/Read/ReadVariableOp"conv3d_17/bias/Read/ReadVariableOp$conv3d_22/kernel/Read/ReadVariableOp"conv3d_22/bias/Read/ReadVariableOp$conv3d_27/kernel/Read/ReadVariableOp"conv3d_27/bias/Read/ReadVariableOp$conv3d_18/kernel/Read/ReadVariableOp"conv3d_18/bias/Read/ReadVariableOp$conv3d_23/kernel/Read/ReadVariableOp"conv3d_23/bias/Read/ReadVariableOp$conv3d_28/kernel/Read/ReadVariableOp"conv3d_28/bias/Read/ReadVariableOp$conv3d_19/kernel/Read/ReadVariableOp"conv3d_19/bias/Read/ReadVariableOp$conv3d_24/kernel/Read/ReadVariableOp"conv3d_24/bias/Read/ReadVariableOp$conv3d_29/kernel/Read/ReadVariableOp"conv3d_29/bias/Read/ReadVariableOp$conv3d_30/kernel/Read/ReadVariableOp"conv3d_30/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp+Adam/conv3d_10/kernel/m/Read/ReadVariableOp)Adam/conv3d_10/bias/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp*Adam/conv3d_6/kernel/m/Read/ReadVariableOp(Adam/conv3d_6/bias/m/Read/ReadVariableOp+Adam/conv3d_11/kernel/m/Read/ReadVariableOp)Adam/conv3d_11/bias/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp*Adam/conv3d_7/kernel/m/Read/ReadVariableOp(Adam/conv3d_7/bias/m/Read/ReadVariableOp+Adam/conv3d_12/kernel/m/Read/ReadVariableOp)Adam/conv3d_12/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_8/kernel/m/Read/ReadVariableOp(Adam/conv3d_8/bias/m/Read/ReadVariableOp+Adam/conv3d_13/kernel/m/Read/ReadVariableOp)Adam/conv3d_13/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/conv3d_9/kernel/m/Read/ReadVariableOp(Adam/conv3d_9/bias/m/Read/ReadVariableOp+Adam/conv3d_14/kernel/m/Read/ReadVariableOp)Adam/conv3d_14/bias/m/Read/ReadVariableOp+Adam/conv3d_15/kernel/m/Read/ReadVariableOp)Adam/conv3d_15/bias/m/Read/ReadVariableOp+Adam/conv3d_20/kernel/m/Read/ReadVariableOp)Adam/conv3d_20/bias/m/Read/ReadVariableOp+Adam/conv3d_25/kernel/m/Read/ReadVariableOp)Adam/conv3d_25/bias/m/Read/ReadVariableOp+Adam/conv3d_16/kernel/m/Read/ReadVariableOp)Adam/conv3d_16/bias/m/Read/ReadVariableOp+Adam/conv3d_21/kernel/m/Read/ReadVariableOp)Adam/conv3d_21/bias/m/Read/ReadVariableOp+Adam/conv3d_26/kernel/m/Read/ReadVariableOp)Adam/conv3d_26/bias/m/Read/ReadVariableOp+Adam/conv3d_17/kernel/m/Read/ReadVariableOp)Adam/conv3d_17/bias/m/Read/ReadVariableOp+Adam/conv3d_22/kernel/m/Read/ReadVariableOp)Adam/conv3d_22/bias/m/Read/ReadVariableOp+Adam/conv3d_27/kernel/m/Read/ReadVariableOp)Adam/conv3d_27/bias/m/Read/ReadVariableOp+Adam/conv3d_18/kernel/m/Read/ReadVariableOp)Adam/conv3d_18/bias/m/Read/ReadVariableOp+Adam/conv3d_23/kernel/m/Read/ReadVariableOp)Adam/conv3d_23/bias/m/Read/ReadVariableOp+Adam/conv3d_28/kernel/m/Read/ReadVariableOp)Adam/conv3d_28/bias/m/Read/ReadVariableOp+Adam/conv3d_19/kernel/m/Read/ReadVariableOp)Adam/conv3d_19/bias/m/Read/ReadVariableOp+Adam/conv3d_24/kernel/m/Read/ReadVariableOp)Adam/conv3d_24/bias/m/Read/ReadVariableOp+Adam/conv3d_29/kernel/m/Read/ReadVariableOp)Adam/conv3d_29/bias/m/Read/ReadVariableOp+Adam/conv3d_30/kernel/m/Read/ReadVariableOp)Adam/conv3d_30/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp+Adam/conv3d_10/kernel/v/Read/ReadVariableOp)Adam/conv3d_10/bias/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp*Adam/conv3d_6/kernel/v/Read/ReadVariableOp(Adam/conv3d_6/bias/v/Read/ReadVariableOp+Adam/conv3d_11/kernel/v/Read/ReadVariableOp)Adam/conv3d_11/bias/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp*Adam/conv3d_7/kernel/v/Read/ReadVariableOp(Adam/conv3d_7/bias/v/Read/ReadVariableOp+Adam/conv3d_12/kernel/v/Read/ReadVariableOp)Adam/conv3d_12/bias/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_8/kernel/v/Read/ReadVariableOp(Adam/conv3d_8/bias/v/Read/ReadVariableOp+Adam/conv3d_13/kernel/v/Read/ReadVariableOp)Adam/conv3d_13/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/conv3d_9/kernel/v/Read/ReadVariableOp(Adam/conv3d_9/bias/v/Read/ReadVariableOp+Adam/conv3d_14/kernel/v/Read/ReadVariableOp)Adam/conv3d_14/bias/v/Read/ReadVariableOp+Adam/conv3d_15/kernel/v/Read/ReadVariableOp)Adam/conv3d_15/bias/v/Read/ReadVariableOp+Adam/conv3d_20/kernel/v/Read/ReadVariableOp)Adam/conv3d_20/bias/v/Read/ReadVariableOp+Adam/conv3d_25/kernel/v/Read/ReadVariableOp)Adam/conv3d_25/bias/v/Read/ReadVariableOp+Adam/conv3d_16/kernel/v/Read/ReadVariableOp)Adam/conv3d_16/bias/v/Read/ReadVariableOp+Adam/conv3d_21/kernel/v/Read/ReadVariableOp)Adam/conv3d_21/bias/v/Read/ReadVariableOp+Adam/conv3d_26/kernel/v/Read/ReadVariableOp)Adam/conv3d_26/bias/v/Read/ReadVariableOp+Adam/conv3d_17/kernel/v/Read/ReadVariableOp)Adam/conv3d_17/bias/v/Read/ReadVariableOp+Adam/conv3d_22/kernel/v/Read/ReadVariableOp)Adam/conv3d_22/bias/v/Read/ReadVariableOp+Adam/conv3d_27/kernel/v/Read/ReadVariableOp)Adam/conv3d_27/bias/v/Read/ReadVariableOp+Adam/conv3d_18/kernel/v/Read/ReadVariableOp)Adam/conv3d_18/bias/v/Read/ReadVariableOp+Adam/conv3d_23/kernel/v/Read/ReadVariableOp)Adam/conv3d_23/bias/v/Read/ReadVariableOp+Adam/conv3d_28/kernel/v/Read/ReadVariableOp)Adam/conv3d_28/bias/v/Read/ReadVariableOp+Adam/conv3d_19/kernel/v/Read/ReadVariableOp)Adam/conv3d_19/bias/v/Read/ReadVariableOp+Adam/conv3d_24/kernel/v/Read/ReadVariableOp)Adam/conv3d_24/bias/v/Read/ReadVariableOp+Adam/conv3d_29/kernel/v/Read/ReadVariableOp)Adam/conv3d_29/bias/v/Read/ReadVariableOp+Adam/conv3d_30/kernel/v/Read/ReadVariableOp)Adam/conv3d_30/bias/v/Read/ReadVariableOpConst*б
TinЩ
Ц2У	*
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
GPU2 *0J 8 *(
f#R!
__inference__traced_save_137335
Љ#
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_5/kernelconv3d_5/biasconv3d_10/kernelconv3d_10/biasconv3d_1/kernelconv3d_1/biasconv3d_6/kernelconv3d_6/biasconv3d_11/kernelconv3d_11/biasconv3d_2/kernelconv3d_2/biasconv3d_7/kernelconv3d_7/biasconv3d_12/kernelconv3d_12/biasconv3d_3/kernelconv3d_3/biasconv3d_8/kernelconv3d_8/biasconv3d_13/kernelconv3d_13/biasconv3d_4/kernelconv3d_4/biasconv3d_9/kernelconv3d_9/biasconv3d_14/kernelconv3d_14/biasconv3d_15/kernelconv3d_15/biasconv3d_20/kernelconv3d_20/biasconv3d_25/kernelconv3d_25/biasconv3d_16/kernelconv3d_16/biasconv3d_21/kernelconv3d_21/biasconv3d_26/kernelconv3d_26/biasconv3d_17/kernelconv3d_17/biasconv3d_22/kernelconv3d_22/biasconv3d_27/kernelconv3d_27/biasconv3d_18/kernelconv3d_18/biasconv3d_23/kernelconv3d_23/biasconv3d_28/kernelconv3d_28/biasconv3d_19/kernelconv3d_19/biasconv3d_24/kernelconv3d_24/biasconv3d_29/kernelconv3d_29/biasconv3d_30/kernelconv3d_30/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/mAdam/conv3d_10/kernel/mAdam/conv3d_10/bias/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/mAdam/conv3d_6/kernel/mAdam/conv3d_6/bias/mAdam/conv3d_11/kernel/mAdam/conv3d_11/bias/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/mAdam/conv3d_7/kernel/mAdam/conv3d_7/bias/mAdam/conv3d_12/kernel/mAdam/conv3d_12/bias/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_8/kernel/mAdam/conv3d_8/bias/mAdam/conv3d_13/kernel/mAdam/conv3d_13/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/conv3d_9/kernel/mAdam/conv3d_9/bias/mAdam/conv3d_14/kernel/mAdam/conv3d_14/bias/mAdam/conv3d_15/kernel/mAdam/conv3d_15/bias/mAdam/conv3d_20/kernel/mAdam/conv3d_20/bias/mAdam/conv3d_25/kernel/mAdam/conv3d_25/bias/mAdam/conv3d_16/kernel/mAdam/conv3d_16/bias/mAdam/conv3d_21/kernel/mAdam/conv3d_21/bias/mAdam/conv3d_26/kernel/mAdam/conv3d_26/bias/mAdam/conv3d_17/kernel/mAdam/conv3d_17/bias/mAdam/conv3d_22/kernel/mAdam/conv3d_22/bias/mAdam/conv3d_27/kernel/mAdam/conv3d_27/bias/mAdam/conv3d_18/kernel/mAdam/conv3d_18/bias/mAdam/conv3d_23/kernel/mAdam/conv3d_23/bias/mAdam/conv3d_28/kernel/mAdam/conv3d_28/bias/mAdam/conv3d_19/kernel/mAdam/conv3d_19/bias/mAdam/conv3d_24/kernel/mAdam/conv3d_24/bias/mAdam/conv3d_29/kernel/mAdam/conv3d_29/bias/mAdam/conv3d_30/kernel/mAdam/conv3d_30/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/vAdam/conv3d_10/kernel/vAdam/conv3d_10/bias/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/vAdam/conv3d_6/kernel/vAdam/conv3d_6/bias/vAdam/conv3d_11/kernel/vAdam/conv3d_11/bias/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/vAdam/conv3d_7/kernel/vAdam/conv3d_7/bias/vAdam/conv3d_12/kernel/vAdam/conv3d_12/bias/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_8/kernel/vAdam/conv3d_8/bias/vAdam/conv3d_13/kernel/vAdam/conv3d_13/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/conv3d_9/kernel/vAdam/conv3d_9/bias/vAdam/conv3d_14/kernel/vAdam/conv3d_14/bias/vAdam/conv3d_15/kernel/vAdam/conv3d_15/bias/vAdam/conv3d_20/kernel/vAdam/conv3d_20/bias/vAdam/conv3d_25/kernel/vAdam/conv3d_25/bias/vAdam/conv3d_16/kernel/vAdam/conv3d_16/bias/vAdam/conv3d_21/kernel/vAdam/conv3d_21/bias/vAdam/conv3d_26/kernel/vAdam/conv3d_26/bias/vAdam/conv3d_17/kernel/vAdam/conv3d_17/bias/vAdam/conv3d_22/kernel/vAdam/conv3d_22/bias/vAdam/conv3d_27/kernel/vAdam/conv3d_27/bias/vAdam/conv3d_18/kernel/vAdam/conv3d_18/bias/vAdam/conv3d_23/kernel/vAdam/conv3d_23/bias/vAdam/conv3d_28/kernel/vAdam/conv3d_28/bias/vAdam/conv3d_19/kernel/vAdam/conv3d_19/bias/vAdam/conv3d_24/kernel/vAdam/conv3d_24/bias/vAdam/conv3d_29/kernel/vAdam/conv3d_29/bias/vAdam/conv3d_30/kernel/vAdam/conv3d_30/bias/v*а
TinШ
Х2Т*
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
GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_137924И-
ь
g
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_136259

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_26_layer_call_fn_136308

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_26_layer_call_and_return_conditional_losses_132111{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_29_layer_call_and_return_conditional_losses_132487

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
ќ	
a
E__inference_reshape_6_layer_call_and_return_conditional_losses_136053

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
м
J
.__inference_up_sampling3d_layer_call_fn_136138

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_132098l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
р
L
0__inference_up_sampling3d_5_layer_call_fn_136516

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_132263l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131718

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
юо
ф
A__inference_model_layer_call_and_return_conditional_losses_132520

inputs
inputs_1
inputs_2.
conv3d_10_131612:
conv3d_10_131614:-
conv3d_5_131629:
conv3d_5_131631:+
conv3d_131646:
conv3d_131648:.
conv3d_11_131663:
conv3d_11_131665:-
conv3d_6_131680:
conv3d_6_131682:-
conv3d_1_131697:
conv3d_1_131699:.
conv3d_12_131732:
conv3d_12_131734:-
conv3d_7_131749:
conv3d_7_131751:-
conv3d_2_131766:
conv3d_2_131768:.
conv3d_13_131783:
conv3d_13_131785:-
conv3d_8_131800:
conv3d_8_131802:-
conv3d_3_131817:
conv3d_3_131819:-
conv3d_4_131852:
conv3d_4_131854:-
conv3d_9_131869:
conv3d_9_131871:.
conv3d_14_131886:
conv3d_14_131888:.
conv3d_25_131944:
conv3d_25_131946:.
conv3d_20_131961:
conv3d_20_131963:.
conv3d_15_131978:
conv3d_15_131980:.
conv3d_26_132112:
conv3d_26_132114:.
conv3d_21_132129:
conv3d_21_132131:.
conv3d_16_132146:
conv3d_16_132148:.
conv3d_27_132163:
conv3d_27_132165:.
conv3d_22_132180:
conv3d_22_132182:.
conv3d_17_132197:
conv3d_17_132199:.
conv3d_28_132403:
conv3d_28_132405:.
conv3d_23_132420:
conv3d_23_132422:.
conv3d_18_132437:
conv3d_18_132439:.
conv3d_19_132454:
conv3d_19_132456:.
conv3d_24_132471:
conv3d_24_132473:.
conv3d_29_132488:
conv3d_29_132490:.
conv3d_30_132514:
conv3d_30_132516:
identityЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ!conv3d_10/StatefulPartitionedCallЂ!conv3d_11/StatefulPartitionedCallЂ!conv3d_12/StatefulPartitionedCallЂ!conv3d_13/StatefulPartitionedCallЂ!conv3d_14/StatefulPartitionedCallЂ!conv3d_15/StatefulPartitionedCallЂ!conv3d_16/StatefulPartitionedCallЂ!conv3d_17/StatefulPartitionedCallЂ!conv3d_18/StatefulPartitionedCallЂ!conv3d_19/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂ!conv3d_20/StatefulPartitionedCallЂ!conv3d_21/StatefulPartitionedCallЂ!conv3d_22/StatefulPartitionedCallЂ!conv3d_23/StatefulPartitionedCallЂ!conv3d_24/StatefulPartitionedCallЂ!conv3d_25/StatefulPartitionedCallЂ!conv3d_26/StatefulPartitionedCallЂ!conv3d_27/StatefulPartitionedCallЂ!conv3d_28/StatefulPartitionedCallЂ!conv3d_29/StatefulPartitionedCallЂ conv3d_3/StatefulPartitionedCallЂ!conv3d_30/StatefulPartitionedCallЂ conv3d_4/StatefulPartitionedCallЂ conv3d_5/StatefulPartitionedCallЂ conv3d_6/StatefulPartitionedCallЂ conv3d_7/StatefulPartitionedCallЂ conv3d_8/StatefulPartitionedCallЂ conv3d_9/StatefulPartitionedCallЧ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_131554Э
reshape_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_131571Э
reshape_2/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_131588Г
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_131598Ѓ
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_10_131612conv3d_10_131614*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_131611
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_5_131629conv3d_5_131631*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_131628
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_131646conv3d_131648*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_131645Љ
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_131663conv3d_11_131665*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_131662Є
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_131680conv3d_6_131682*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_131679Ђ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_131697conv3d_1_131699*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_131696ћ
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131706њ
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131712і
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131718Ї
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_131732conv3d_12_131734*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_131731Ѓ
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_131749conv3d_7_131751*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_131748Ё
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_131766conv3d_2_131768*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_131765Љ
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_131783conv3d_13_131785*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_131782Є
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_131800conv3d_8_131802*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_131799Є
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_131817conv3d_3_131819*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_131816ћ
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131826њ
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131832њ
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131838Ѓ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_131852conv3d_4_131854*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_131851Ѓ
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_131869conv3d_9_131871*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_131868Ї
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_131886conv3d_14_131888*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_131885Л
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_131899ж
reshape_6/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_131913ч
reshape_7/PartitionedCallPartitionedCall"reshape_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_7_layer_call_and_return_conditional_losses_131930Ё
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_25_131944conv3d_25_131946*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_25_layer_call_and_return_conditional_losses_131943Ё
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_20_131961conv3d_20_131963*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_20_layer_call_and_return_conditional_losses_131960Ё
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_15_131978conv3d_15_131980*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_131977ћ
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_132020ћ
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_132059ї
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_132098Ї
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_132112conv3d_26_132114*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_26_layer_call_and_return_conditional_losses_132111Ї
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_132129conv3d_21_132131*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_21_layer_call_and_return_conditional_losses_132128Ѕ
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_132146conv3d_16_132148*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_132145Љ
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_132163conv3d_27_132165*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_27_layer_call_and_return_conditional_losses_132162Љ
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_132180conv3d_22_132182*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_22_layer_call_and_return_conditional_losses_132179Љ
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_132197conv3d_17_132199*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_132196ћ
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_132263ћ
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_132326ћ
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_132389Ї
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_132403conv3d_28_132405*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_28_layer_call_and_return_conditional_losses_132402Ї
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_132420conv3d_23_132422*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_23_layer_call_and_return_conditional_losses_132419Ї
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_132437conv3d_18_132439*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_132436Љ
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_132454conv3d_19_132456*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_132453Љ
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_132471conv3d_24_132473*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_24_layer_call_and_return_conditional_losses_132470Љ
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_132488conv3d_29_132490*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_29_layer_call_and_return_conditional_losses_132487С
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_132501
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_132514conv3d_30_132516*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_30_layer_call_and_return_conditional_losses_132513
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2F
!conv3d_20/StatefulPartitionedCall!conv3d_20/StatefulPartitionedCall2F
!conv3d_21/StatefulPartitionedCall!conv3d_21/StatefulPartitionedCall2F
!conv3d_22/StatefulPartitionedCall!conv3d_22/StatefulPartitionedCall2F
!conv3d_23/StatefulPartitionedCall!conv3d_23/StatefulPartitionedCall2F
!conv3d_24/StatefulPartitionedCall!conv3d_24/StatefulPartitionedCall2F
!conv3d_25/StatefulPartitionedCall!conv3d_25/StatefulPartitionedCall2F
!conv3d_26/StatefulPartitionedCall!conv3d_26/StatefulPartitionedCall2F
!conv3d_27/StatefulPartitionedCall!conv3d_27/StatefulPartitionedCall2F
!conv3d_28/StatefulPartitionedCall!conv3d_28/StatefulPartitionedCall2F
!conv3d_29/StatefulPartitionedCall!conv3d_29/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2F
!conv3d_30/StatefulPartitionedCall!conv3d_30/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_25_layer_call_and_return_conditional_losses_131943

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_135741

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
І

D__inference_conv3d_8_layer_call_and_return_conditional_losses_131799

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
И%
g
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_136445

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :г
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :т
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_17_layer_call_fn_136328

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_132196{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

`
&__inference_add_1_layer_call_fn_136704
inputs_0
inputs_1
inputs_2
identityе
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_132501l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   :] Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/2

Ѓ
*__inference_conv3d_19_layer_call_fn_136646

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_132453{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_20_layer_call_and_return_conditional_losses_136113

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_25_layer_call_and_return_conditional_losses_136133

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
с
6
!__inference__wrapped_model_131456	
u_vel	
v_vel	
w_velL
.model_conv3d_10_conv3d_readvariableop_resource:=
/model_conv3d_10_biasadd_readvariableop_resource:K
-model_conv3d_5_conv3d_readvariableop_resource:<
.model_conv3d_5_biasadd_readvariableop_resource:I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:L
.model_conv3d_11_conv3d_readvariableop_resource:=
/model_conv3d_11_biasadd_readvariableop_resource:K
-model_conv3d_6_conv3d_readvariableop_resource:<
.model_conv3d_6_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource:<
.model_conv3d_1_biasadd_readvariableop_resource:L
.model_conv3d_12_conv3d_readvariableop_resource:=
/model_conv3d_12_biasadd_readvariableop_resource:K
-model_conv3d_7_conv3d_readvariableop_resource:<
.model_conv3d_7_biasadd_readvariableop_resource:K
-model_conv3d_2_conv3d_readvariableop_resource:<
.model_conv3d_2_biasadd_readvariableop_resource:L
.model_conv3d_13_conv3d_readvariableop_resource:=
/model_conv3d_13_biasadd_readvariableop_resource:K
-model_conv3d_8_conv3d_readvariableop_resource:<
.model_conv3d_8_biasadd_readvariableop_resource:K
-model_conv3d_3_conv3d_readvariableop_resource:<
.model_conv3d_3_biasadd_readvariableop_resource:K
-model_conv3d_4_conv3d_readvariableop_resource:<
.model_conv3d_4_biasadd_readvariableop_resource:K
-model_conv3d_9_conv3d_readvariableop_resource:<
.model_conv3d_9_biasadd_readvariableop_resource:L
.model_conv3d_14_conv3d_readvariableop_resource:=
/model_conv3d_14_biasadd_readvariableop_resource:L
.model_conv3d_25_conv3d_readvariableop_resource:=
/model_conv3d_25_biasadd_readvariableop_resource:L
.model_conv3d_20_conv3d_readvariableop_resource:=
/model_conv3d_20_biasadd_readvariableop_resource:L
.model_conv3d_15_conv3d_readvariableop_resource:=
/model_conv3d_15_biasadd_readvariableop_resource:L
.model_conv3d_26_conv3d_readvariableop_resource:=
/model_conv3d_26_biasadd_readvariableop_resource:L
.model_conv3d_21_conv3d_readvariableop_resource:=
/model_conv3d_21_biasadd_readvariableop_resource:L
.model_conv3d_16_conv3d_readvariableop_resource:=
/model_conv3d_16_biasadd_readvariableop_resource:L
.model_conv3d_27_conv3d_readvariableop_resource:=
/model_conv3d_27_biasadd_readvariableop_resource:L
.model_conv3d_22_conv3d_readvariableop_resource:=
/model_conv3d_22_biasadd_readvariableop_resource:L
.model_conv3d_17_conv3d_readvariableop_resource:=
/model_conv3d_17_biasadd_readvariableop_resource:L
.model_conv3d_28_conv3d_readvariableop_resource:=
/model_conv3d_28_biasadd_readvariableop_resource:L
.model_conv3d_23_conv3d_readvariableop_resource:=
/model_conv3d_23_biasadd_readvariableop_resource:L
.model_conv3d_18_conv3d_readvariableop_resource:=
/model_conv3d_18_biasadd_readvariableop_resource:L
.model_conv3d_19_conv3d_readvariableop_resource:=
/model_conv3d_19_biasadd_readvariableop_resource:L
.model_conv3d_24_conv3d_readvariableop_resource:=
/model_conv3d_24_biasadd_readvariableop_resource:L
.model_conv3d_29_conv3d_readvariableop_resource:=
/model_conv3d_29_biasadd_readvariableop_resource:L
.model_conv3d_30_conv3d_readvariableop_resource:=
/model_conv3d_30_biasadd_readvariableop_resource:
identityЂ#model/conv3d/BiasAdd/ReadVariableOpЂ"model/conv3d/Conv3D/ReadVariableOpЂ%model/conv3d_1/BiasAdd/ReadVariableOpЂ$model/conv3d_1/Conv3D/ReadVariableOpЂ&model/conv3d_10/BiasAdd/ReadVariableOpЂ%model/conv3d_10/Conv3D/ReadVariableOpЂ&model/conv3d_11/BiasAdd/ReadVariableOpЂ%model/conv3d_11/Conv3D/ReadVariableOpЂ&model/conv3d_12/BiasAdd/ReadVariableOpЂ%model/conv3d_12/Conv3D/ReadVariableOpЂ&model/conv3d_13/BiasAdd/ReadVariableOpЂ%model/conv3d_13/Conv3D/ReadVariableOpЂ&model/conv3d_14/BiasAdd/ReadVariableOpЂ%model/conv3d_14/Conv3D/ReadVariableOpЂ&model/conv3d_15/BiasAdd/ReadVariableOpЂ%model/conv3d_15/Conv3D/ReadVariableOpЂ&model/conv3d_16/BiasAdd/ReadVariableOpЂ%model/conv3d_16/Conv3D/ReadVariableOpЂ&model/conv3d_17/BiasAdd/ReadVariableOpЂ%model/conv3d_17/Conv3D/ReadVariableOpЂ&model/conv3d_18/BiasAdd/ReadVariableOpЂ%model/conv3d_18/Conv3D/ReadVariableOpЂ&model/conv3d_19/BiasAdd/ReadVariableOpЂ%model/conv3d_19/Conv3D/ReadVariableOpЂ%model/conv3d_2/BiasAdd/ReadVariableOpЂ$model/conv3d_2/Conv3D/ReadVariableOpЂ&model/conv3d_20/BiasAdd/ReadVariableOpЂ%model/conv3d_20/Conv3D/ReadVariableOpЂ&model/conv3d_21/BiasAdd/ReadVariableOpЂ%model/conv3d_21/Conv3D/ReadVariableOpЂ&model/conv3d_22/BiasAdd/ReadVariableOpЂ%model/conv3d_22/Conv3D/ReadVariableOpЂ&model/conv3d_23/BiasAdd/ReadVariableOpЂ%model/conv3d_23/Conv3D/ReadVariableOpЂ&model/conv3d_24/BiasAdd/ReadVariableOpЂ%model/conv3d_24/Conv3D/ReadVariableOpЂ&model/conv3d_25/BiasAdd/ReadVariableOpЂ%model/conv3d_25/Conv3D/ReadVariableOpЂ&model/conv3d_26/BiasAdd/ReadVariableOpЂ%model/conv3d_26/Conv3D/ReadVariableOpЂ&model/conv3d_27/BiasAdd/ReadVariableOpЂ%model/conv3d_27/Conv3D/ReadVariableOpЂ&model/conv3d_28/BiasAdd/ReadVariableOpЂ%model/conv3d_28/Conv3D/ReadVariableOpЂ&model/conv3d_29/BiasAdd/ReadVariableOpЂ%model/conv3d_29/Conv3D/ReadVariableOpЂ%model/conv3d_3/BiasAdd/ReadVariableOpЂ$model/conv3d_3/Conv3D/ReadVariableOpЂ&model/conv3d_30/BiasAdd/ReadVariableOpЂ%model/conv3d_30/Conv3D/ReadVariableOpЂ%model/conv3d_4/BiasAdd/ReadVariableOpЂ$model/conv3d_4/Conv3D/ReadVariableOpЂ%model/conv3d_5/BiasAdd/ReadVariableOpЂ$model/conv3d_5/Conv3D/ReadVariableOpЂ%model/conv3d_6/BiasAdd/ReadVariableOpЂ$model/conv3d_6/Conv3D/ReadVariableOpЂ%model/conv3d_7/BiasAdd/ReadVariableOpЂ$model/conv3d_7/Conv3D/ReadVariableOpЂ%model/conv3d_8/BiasAdd/ReadVariableOpЂ$model/conv3d_8/Conv3D/ReadVariableOpЂ%model/conv3d_9/BiasAdd/ReadVariableOpЂ$model/conv3d_9/Conv3D/ReadVariableOpH
model/reshape/ShapeShapeu_vel*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : _
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : _
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : _
model/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0&model/reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
model/reshape/ReshapeReshapeu_vel$model/reshape/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   J
model/reshape_1/ShapeShapev_vel*
T0*
_output_shapes
:m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : a
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : a
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : a
model/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0(model/reshape_1/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
model/reshape_1/ReshapeReshapev_vel&model/reshape_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   J
model/reshape_2/ShapeShapew_vel*
T0*
_output_shapes
:m
#model/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
model/reshape_2/strided_sliceStridedSlicemodel/reshape_2/Shape:output:0,model/reshape_2/strided_slice/stack:output:0.model/reshape_2/strided_slice/stack_1:output:0.model/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : a
model/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : a
model/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : a
model/reshape_2/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0(model/reshape_2/Reshape/shape/2:output:0(model/reshape_2/Reshape/shape/3:output:0(model/reshape_2/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
model/reshape_2/ReshapeReshapew_vel&model/reshape_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ї
model/concatenate/concatConcatV2model/reshape/Reshape:output:0 model/reshape_1/Reshape:output:0 model/reshape_2/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_10/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0й
model/conv3d_10/Conv3DConv3D!model/concatenate/concat:output:0-model/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_10/BiasAddBiasAddmodel/conv3d_10/Conv3D:output:0.model/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_10/ReluRelu model/conv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0з
model/conv3d_5/Conv3DConv3D!model/concatenate/concat:output:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   z
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0г
model/conv3d/Conv3DConv3D!model/concatenate/concat:output:0*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ј
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_11/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_11/Conv3DConv3D"model/conv3d_10/Relu:activations:0-model/conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_11/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_11/BiasAddBiasAddmodel/conv3d_11/Conv3D:output:0.model/conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_11/ReluRelu model/conv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0з
model/conv3d_6/Conv3DConv3D!model/conv3d_5/Relu:activations:0,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   z
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0е
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   Ы
model/max_pooling3d_4/MaxPool3D	MaxPool3D"model/conv3d_11/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
Ъ
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
Ш
model/max_pooling3d/MaxPool3D	MaxPool3D!model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
 
%model/conv3d_12/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0р
model/conv3d_12/Conv3DConv3D(model/max_pooling3d_4/MaxPool3D:output:0-model/conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_12/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_12/BiasAddBiasAddmodel/conv3d_12/Conv3D:output:0.model/conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_12/ReluRelu model/conv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0о
model/conv3d_7/Conv3DConv3D(model/max_pooling3d_2/MaxPool3D:output:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0м
model/conv3d_2/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_13/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_13/Conv3DConv3D"model/conv3d_12/Relu:activations:0-model/conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_13/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_13/BiasAddBiasAddmodel/conv3d_13/Conv3D:output:0.model/conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_13/ReluRelu model/conv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
$model/conv3d_8/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0з
model/conv3d_8/Conv3DConv3D!model/conv3d_7/Relu:activations:0,model/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_8/BiasAddBiasAddmodel/conv3d_8/Conv3D:output:0-model/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_8/ReluRelumodel/conv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0з
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЫ
model/max_pooling3d_5/MaxPool3D	MaxPool3D"model/conv3d_13/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
Ъ
model/max_pooling3d_3/MaxPool3D	MaxPool3D!model/conv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
Ъ
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0о
model/conv3d_4/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_4/TanhTanhmodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
$model/conv3d_9/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0о
model/conv3d_9/Conv3DConv3D(model/max_pooling3d_3/MaxPool3D:output:0,model/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_9/BiasAddBiasAddmodel/conv3d_9/Conv3D:output:0-model/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_9/TanhTanhmodel/conv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_14/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0р
model/conv3d_14/Conv3DConv3D(model/max_pooling3d_5/MaxPool3D:output:0-model/conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_14/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_14/BiasAddBiasAddmodel/conv3d_14/Conv3D:output:0.model/conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_14/TanhTanh model/conv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
model/add/addAddV2model/conv3d_4/Tanh:y:0model/conv3d_9/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ
model/add/add_1AddV2model/add/add:z:0model/conv3d_14/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџX
model/reshape_6/ShapeShapemodel/add/add_1:z:0*
T0*
_output_shapes
:m
#model/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
model/reshape_6/strided_sliceStridedSlicemodel/reshape_6/Shape:output:0,model/reshape_6/strided_slice/stack:output:0.model/reshape_6/strided_slice/stack_1:output:0.model/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Ѕ
model/reshape_6/Reshape/shapePack&model/reshape_6/strided_slice:output:0(model/reshape_6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
model/reshape_6/ReshapeReshapemodel/add/add_1:z:0&model/reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџe
model/reshape_7/ShapeShape model/reshape_6/Reshape:output:0*
T0*
_output_shapes
:m
#model/reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
model/reshape_7/strided_sliceStridedSlicemodel/reshape_7/Shape:output:0,model/reshape_7/strided_slice/stack:output:0.model/reshape_7/strided_slice/stack_1:output:0.model/reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_7/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
model/reshape_7/Reshape/shapePack&model/reshape_7/strided_slice:output:0(model/reshape_7/Reshape/shape/1:output:0(model/reshape_7/Reshape/shape/2:output:0(model/reshape_7/Reshape/shape/3:output:0(model/reshape_7/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:Њ
model/reshape_7/ReshapeReshape model/reshape_6/Reshape:output:0&model/reshape_7/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_25/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_25_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0и
model/conv3d_25/Conv3DConv3D model/reshape_7/Reshape:output:0-model/conv3d_25/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_25/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_25/BiasAddBiasAddmodel/conv3d_25/Conv3D:output:0.model/conv3d_25/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_25/ReluRelu model/conv3d_25/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_20/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_20_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0и
model/conv3d_20/Conv3DConv3D model/reshape_7/Reshape:output:0-model/conv3d_20/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_20/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_20/BiasAddBiasAddmodel/conv3d_20/Conv3D:output:0.model/conv3d_20/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_20/ReluRelu model/conv3d_20/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_15/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0и
model/conv3d_15/Conv3DConv3D model/reshape_7/Reshape:output:0-model/conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_15/BiasAddBiasAddmodel/conv3d_15/Conv3D:output:0.model/conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_15/ReluRelu model/conv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
model/up_sampling3d_4/splitSplit.model/up_sampling3d_4/split/split_dim:output:0"model/conv3d_25/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
model/up_sampling3d_4/concatConcatV2$model/up_sampling3d_4/split:output:0$model/up_sampling3d_4/split:output:0$model/up_sampling3d_4/split:output:1$model/up_sampling3d_4/split:output:1$model/up_sampling3d_4/split:output:2$model/up_sampling3d_4/split:output:2$model/up_sampling3d_4/split:output:3$model/up_sampling3d_4/split:output:3$model/up_sampling3d_4/split:output:4$model/up_sampling3d_4/split:output:4$model/up_sampling3d_4/split:output:5$model/up_sampling3d_4/split:output:5$model/up_sampling3d_4/split:output:6$model/up_sampling3d_4/split:output:6$model/up_sampling3d_4/split:output:7$model/up_sampling3d_4/split:output:7*model/up_sampling3d_4/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџi
'model/up_sampling3d_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
model/up_sampling3d_4/split_1Split0model/up_sampling3d_4/split_1/split_dim:output:0%model/up_sampling3d_4/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splite
#model/up_sampling3d_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_4/concat_1ConcatV2&model/up_sampling3d_4/split_1:output:0&model/up_sampling3d_4/split_1:output:0&model/up_sampling3d_4/split_1:output:1&model/up_sampling3d_4/split_1:output:1&model/up_sampling3d_4/split_1:output:2&model/up_sampling3d_4/split_1:output:2&model/up_sampling3d_4/split_1:output:3&model/up_sampling3d_4/split_1:output:3&model/up_sampling3d_4/split_1:output:4&model/up_sampling3d_4/split_1:output:4&model/up_sampling3d_4/split_1:output:5&model/up_sampling3d_4/split_1:output:5&model/up_sampling3d_4/split_1:output:6&model/up_sampling3d_4/split_1:output:6&model/up_sampling3d_4/split_1:output:7&model/up_sampling3d_4/split_1:output:7,model/up_sampling3d_4/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџi
'model/up_sampling3d_4/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ќ
model/up_sampling3d_4/split_2Split0model/up_sampling3d_4/split_2/split_dim:output:0'model/up_sampling3d_4/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splite
#model/up_sampling3d_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_4/concat_2ConcatV2&model/up_sampling3d_4/split_2:output:0&model/up_sampling3d_4/split_2:output:0&model/up_sampling3d_4/split_2:output:1&model/up_sampling3d_4/split_2:output:1&model/up_sampling3d_4/split_2:output:2&model/up_sampling3d_4/split_2:output:2&model/up_sampling3d_4/split_2:output:3&model/up_sampling3d_4/split_2:output:3&model/up_sampling3d_4/split_2:output:4&model/up_sampling3d_4/split_2:output:4&model/up_sampling3d_4/split_2:output:5&model/up_sampling3d_4/split_2:output:5&model/up_sampling3d_4/split_2:output:6&model/up_sampling3d_4/split_2:output:6&model/up_sampling3d_4/split_2:output:7&model/up_sampling3d_4/split_2:output:7,model/up_sampling3d_4/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
model/up_sampling3d_2/splitSplit.model/up_sampling3d_2/split/split_dim:output:0"model/conv3d_20/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
model/up_sampling3d_2/concatConcatV2$model/up_sampling3d_2/split:output:0$model/up_sampling3d_2/split:output:0$model/up_sampling3d_2/split:output:1$model/up_sampling3d_2/split:output:1$model/up_sampling3d_2/split:output:2$model/up_sampling3d_2/split:output:2$model/up_sampling3d_2/split:output:3$model/up_sampling3d_2/split:output:3$model/up_sampling3d_2/split:output:4$model/up_sampling3d_2/split:output:4$model/up_sampling3d_2/split:output:5$model/up_sampling3d_2/split:output:5$model/up_sampling3d_2/split:output:6$model/up_sampling3d_2/split:output:6$model/up_sampling3d_2/split:output:7$model/up_sampling3d_2/split:output:7*model/up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџi
'model/up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
model/up_sampling3d_2/split_1Split0model/up_sampling3d_2/split_1/split_dim:output:0%model/up_sampling3d_2/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splite
#model/up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_2/concat_1ConcatV2&model/up_sampling3d_2/split_1:output:0&model/up_sampling3d_2/split_1:output:0&model/up_sampling3d_2/split_1:output:1&model/up_sampling3d_2/split_1:output:1&model/up_sampling3d_2/split_1:output:2&model/up_sampling3d_2/split_1:output:2&model/up_sampling3d_2/split_1:output:3&model/up_sampling3d_2/split_1:output:3&model/up_sampling3d_2/split_1:output:4&model/up_sampling3d_2/split_1:output:4&model/up_sampling3d_2/split_1:output:5&model/up_sampling3d_2/split_1:output:5&model/up_sampling3d_2/split_1:output:6&model/up_sampling3d_2/split_1:output:6&model/up_sampling3d_2/split_1:output:7&model/up_sampling3d_2/split_1:output:7,model/up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџi
'model/up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ќ
model/up_sampling3d_2/split_2Split0model/up_sampling3d_2/split_2/split_dim:output:0'model/up_sampling3d_2/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splite
#model/up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_2/concat_2ConcatV2&model/up_sampling3d_2/split_2:output:0&model/up_sampling3d_2/split_2:output:0&model/up_sampling3d_2/split_2:output:1&model/up_sampling3d_2/split_2:output:1&model/up_sampling3d_2/split_2:output:2&model/up_sampling3d_2/split_2:output:2&model/up_sampling3d_2/split_2:output:3&model/up_sampling3d_2/split_2:output:3&model/up_sampling3d_2/split_2:output:4&model/up_sampling3d_2/split_2:output:4&model/up_sampling3d_2/split_2:output:5&model/up_sampling3d_2/split_2:output:5&model/up_sampling3d_2/split_2:output:6&model/up_sampling3d_2/split_2:output:6&model/up_sampling3d_2/split_2:output:7&model/up_sampling3d_2/split_2:output:7,model/up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
#model/up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d/splitSplit,model/up_sampling3d/split/split_dim:output:0"model/conv3d_15/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splita
model/up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :з
model/up_sampling3d/concatConcatV2"model/up_sampling3d/split:output:0"model/up_sampling3d/split:output:0"model/up_sampling3d/split:output:1"model/up_sampling3d/split:output:1"model/up_sampling3d/split:output:2"model/up_sampling3d/split:output:2"model/up_sampling3d/split:output:3"model/up_sampling3d/split:output:3"model/up_sampling3d/split:output:4"model/up_sampling3d/split:output:4"model/up_sampling3d/split:output:5"model/up_sampling3d/split:output:5"model/up_sampling3d/split:output:6"model/up_sampling3d/split:output:6"model/up_sampling3d/split:output:7"model/up_sampling3d/split:output:7(model/up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
model/up_sampling3d/split_1Split.model/up_sampling3d/split_1/split_dim:output:0#model/up_sampling3d/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
model/up_sampling3d/concat_1ConcatV2$model/up_sampling3d/split_1:output:0$model/up_sampling3d/split_1:output:0$model/up_sampling3d/split_1:output:1$model/up_sampling3d/split_1:output:1$model/up_sampling3d/split_1:output:2$model/up_sampling3d/split_1:output:2$model/up_sampling3d/split_1:output:3$model/up_sampling3d/split_1:output:3$model/up_sampling3d/split_1:output:4$model/up_sampling3d/split_1:output:4$model/up_sampling3d/split_1:output:5$model/up_sampling3d/split_1:output:5$model/up_sampling3d/split_1:output:6$model/up_sampling3d/split_1:output:6$model/up_sampling3d/split_1:output:7$model/up_sampling3d/split_1:output:7*model/up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :І
model/up_sampling3d/split_2Split.model/up_sampling3d/split_2/split_dim:output:0%model/up_sampling3d/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
model/up_sampling3d/concat_2ConcatV2$model/up_sampling3d/split_2:output:0$model/up_sampling3d/split_2:output:0$model/up_sampling3d/split_2:output:1$model/up_sampling3d/split_2:output:1$model/up_sampling3d/split_2:output:2$model/up_sampling3d/split_2:output:2$model/up_sampling3d/split_2:output:3$model/up_sampling3d/split_2:output:3$model/up_sampling3d/split_2:output:4$model/up_sampling3d/split_2:output:4$model/up_sampling3d/split_2:output:5$model/up_sampling3d/split_2:output:5$model/up_sampling3d/split_2:output:6$model/up_sampling3d/split_2:output:6$model/up_sampling3d/split_2:output:7$model/up_sampling3d/split_2:output:7*model/up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_26/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_26_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0п
model/conv3d_26/Conv3DConv3D'model/up_sampling3d_4/concat_2:output:0-model/conv3d_26/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_26/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_26/BiasAddBiasAddmodel/conv3d_26/Conv3D:output:0.model/conv3d_26/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_26/ReluRelu model/conv3d_26/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_21/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_21_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0п
model/conv3d_21/Conv3DConv3D'model/up_sampling3d_2/concat_2:output:0-model/conv3d_21/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_21/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_21/BiasAddBiasAddmodel/conv3d_21/Conv3D:output:0.model/conv3d_21/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_21/ReluRelu model/conv3d_21/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_16/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0н
model/conv3d_16/Conv3DConv3D%model/up_sampling3d/concat_2:output:0-model/conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_16/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_16/BiasAddBiasAddmodel/conv3d_16/Conv3D:output:0.model/conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_16/ReluRelu model/conv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_27/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_27_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_27/Conv3DConv3D"model/conv3d_26/Relu:activations:0-model/conv3d_27/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_27/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_27/BiasAddBiasAddmodel/conv3d_27/Conv3D:output:0.model/conv3d_27/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_27/ReluRelu model/conv3d_27/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_22/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_22_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_22/Conv3DConv3D"model/conv3d_21/Relu:activations:0-model/conv3d_22/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_22/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_22/BiasAddBiasAddmodel/conv3d_22/Conv3D:output:0.model/conv3d_22/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_22/ReluRelu model/conv3d_22/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
%model/conv3d_17/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_17/Conv3DConv3D"model/conv3d_16/Relu:activations:0-model/conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_17/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_17/BiasAddBiasAddmodel/conv3d_17/Conv3D:output:0.model/conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_17/ReluRelu model/conv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_5/splitSplit.model/up_sampling3d_5/split/split_dim:output:0"model/conv3d_27/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч

model/up_sampling3d_5/concatConcatV2$model/up_sampling3d_5/split:output:0$model/up_sampling3d_5/split:output:0$model/up_sampling3d_5/split:output:1$model/up_sampling3d_5/split:output:1$model/up_sampling3d_5/split:output:2$model/up_sampling3d_5/split:output:2$model/up_sampling3d_5/split:output:3$model/up_sampling3d_5/split:output:3$model/up_sampling3d_5/split:output:4$model/up_sampling3d_5/split:output:4$model/up_sampling3d_5/split:output:5$model/up_sampling3d_5/split:output:5$model/up_sampling3d_5/split:output:6$model/up_sampling3d_5/split:output:6$model/up_sampling3d_5/split:output:7$model/up_sampling3d_5/split:output:7$model/up_sampling3d_5/split:output:8$model/up_sampling3d_5/split:output:8$model/up_sampling3d_5/split:output:9$model/up_sampling3d_5/split:output:9%model/up_sampling3d_5/split:output:10%model/up_sampling3d_5/split:output:10%model/up_sampling3d_5/split:output:11%model/up_sampling3d_5/split:output:11%model/up_sampling3d_5/split:output:12%model/up_sampling3d_5/split:output:12%model/up_sampling3d_5/split:output:13%model/up_sampling3d_5/split:output:13%model/up_sampling3d_5/split:output:14%model/up_sampling3d_5/split:output:14%model/up_sampling3d_5/split:output:15%model/up_sampling3d_5/split:output:15*model/up_sampling3d_5/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ i
'model/up_sampling3d_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
model/up_sampling3d_5/split_1Split0model/up_sampling3d_5/split_1/split_dim:output:0%model/up_sampling3d_5/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splite
#model/up_sampling3d_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/up_sampling3d_5/concat_1ConcatV2&model/up_sampling3d_5/split_1:output:0&model/up_sampling3d_5/split_1:output:0&model/up_sampling3d_5/split_1:output:1&model/up_sampling3d_5/split_1:output:1&model/up_sampling3d_5/split_1:output:2&model/up_sampling3d_5/split_1:output:2&model/up_sampling3d_5/split_1:output:3&model/up_sampling3d_5/split_1:output:3&model/up_sampling3d_5/split_1:output:4&model/up_sampling3d_5/split_1:output:4&model/up_sampling3d_5/split_1:output:5&model/up_sampling3d_5/split_1:output:5&model/up_sampling3d_5/split_1:output:6&model/up_sampling3d_5/split_1:output:6&model/up_sampling3d_5/split_1:output:7&model/up_sampling3d_5/split_1:output:7&model/up_sampling3d_5/split_1:output:8&model/up_sampling3d_5/split_1:output:8&model/up_sampling3d_5/split_1:output:9&model/up_sampling3d_5/split_1:output:9'model/up_sampling3d_5/split_1:output:10'model/up_sampling3d_5/split_1:output:10'model/up_sampling3d_5/split_1:output:11'model/up_sampling3d_5/split_1:output:11'model/up_sampling3d_5/split_1:output:12'model/up_sampling3d_5/split_1:output:12'model/up_sampling3d_5/split_1:output:13'model/up_sampling3d_5/split_1:output:13'model/up_sampling3d_5/split_1:output:14'model/up_sampling3d_5/split_1:output:14'model/up_sampling3d_5/split_1:output:15'model/up_sampling3d_5/split_1:output:15,model/up_sampling3d_5/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  i
'model/up_sampling3d_5/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
model/up_sampling3d_5/split_2Split0model/up_sampling3d_5/split_2/split_dim:output:0'model/up_sampling3d_5/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splite
#model/up_sampling3d_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/up_sampling3d_5/concat_2ConcatV2&model/up_sampling3d_5/split_2:output:0&model/up_sampling3d_5/split_2:output:0&model/up_sampling3d_5/split_2:output:1&model/up_sampling3d_5/split_2:output:1&model/up_sampling3d_5/split_2:output:2&model/up_sampling3d_5/split_2:output:2&model/up_sampling3d_5/split_2:output:3&model/up_sampling3d_5/split_2:output:3&model/up_sampling3d_5/split_2:output:4&model/up_sampling3d_5/split_2:output:4&model/up_sampling3d_5/split_2:output:5&model/up_sampling3d_5/split_2:output:5&model/up_sampling3d_5/split_2:output:6&model/up_sampling3d_5/split_2:output:6&model/up_sampling3d_5/split_2:output:7&model/up_sampling3d_5/split_2:output:7&model/up_sampling3d_5/split_2:output:8&model/up_sampling3d_5/split_2:output:8&model/up_sampling3d_5/split_2:output:9&model/up_sampling3d_5/split_2:output:9'model/up_sampling3d_5/split_2:output:10'model/up_sampling3d_5/split_2:output:10'model/up_sampling3d_5/split_2:output:11'model/up_sampling3d_5/split_2:output:11'model/up_sampling3d_5/split_2:output:12'model/up_sampling3d_5/split_2:output:12'model/up_sampling3d_5/split_2:output:13'model/up_sampling3d_5/split_2:output:13'model/up_sampling3d_5/split_2:output:14'model/up_sampling3d_5/split_2:output:14'model/up_sampling3d_5/split_2:output:15'model/up_sampling3d_5/split_2:output:15,model/up_sampling3d_5/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   g
%model/up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_3/splitSplit.model/up_sampling3d_3/split/split_dim:output:0"model/conv3d_22/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч

model/up_sampling3d_3/concatConcatV2$model/up_sampling3d_3/split:output:0$model/up_sampling3d_3/split:output:0$model/up_sampling3d_3/split:output:1$model/up_sampling3d_3/split:output:1$model/up_sampling3d_3/split:output:2$model/up_sampling3d_3/split:output:2$model/up_sampling3d_3/split:output:3$model/up_sampling3d_3/split:output:3$model/up_sampling3d_3/split:output:4$model/up_sampling3d_3/split:output:4$model/up_sampling3d_3/split:output:5$model/up_sampling3d_3/split:output:5$model/up_sampling3d_3/split:output:6$model/up_sampling3d_3/split:output:6$model/up_sampling3d_3/split:output:7$model/up_sampling3d_3/split:output:7$model/up_sampling3d_3/split:output:8$model/up_sampling3d_3/split:output:8$model/up_sampling3d_3/split:output:9$model/up_sampling3d_3/split:output:9%model/up_sampling3d_3/split:output:10%model/up_sampling3d_3/split:output:10%model/up_sampling3d_3/split:output:11%model/up_sampling3d_3/split:output:11%model/up_sampling3d_3/split:output:12%model/up_sampling3d_3/split:output:12%model/up_sampling3d_3/split:output:13%model/up_sampling3d_3/split:output:13%model/up_sampling3d_3/split:output:14%model/up_sampling3d_3/split:output:14%model/up_sampling3d_3/split:output:15%model/up_sampling3d_3/split:output:15*model/up_sampling3d_3/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ i
'model/up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
model/up_sampling3d_3/split_1Split0model/up_sampling3d_3/split_1/split_dim:output:0%model/up_sampling3d_3/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splite
#model/up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/up_sampling3d_3/concat_1ConcatV2&model/up_sampling3d_3/split_1:output:0&model/up_sampling3d_3/split_1:output:0&model/up_sampling3d_3/split_1:output:1&model/up_sampling3d_3/split_1:output:1&model/up_sampling3d_3/split_1:output:2&model/up_sampling3d_3/split_1:output:2&model/up_sampling3d_3/split_1:output:3&model/up_sampling3d_3/split_1:output:3&model/up_sampling3d_3/split_1:output:4&model/up_sampling3d_3/split_1:output:4&model/up_sampling3d_3/split_1:output:5&model/up_sampling3d_3/split_1:output:5&model/up_sampling3d_3/split_1:output:6&model/up_sampling3d_3/split_1:output:6&model/up_sampling3d_3/split_1:output:7&model/up_sampling3d_3/split_1:output:7&model/up_sampling3d_3/split_1:output:8&model/up_sampling3d_3/split_1:output:8&model/up_sampling3d_3/split_1:output:9&model/up_sampling3d_3/split_1:output:9'model/up_sampling3d_3/split_1:output:10'model/up_sampling3d_3/split_1:output:10'model/up_sampling3d_3/split_1:output:11'model/up_sampling3d_3/split_1:output:11'model/up_sampling3d_3/split_1:output:12'model/up_sampling3d_3/split_1:output:12'model/up_sampling3d_3/split_1:output:13'model/up_sampling3d_3/split_1:output:13'model/up_sampling3d_3/split_1:output:14'model/up_sampling3d_3/split_1:output:14'model/up_sampling3d_3/split_1:output:15'model/up_sampling3d_3/split_1:output:15,model/up_sampling3d_3/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  i
'model/up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
model/up_sampling3d_3/split_2Split0model/up_sampling3d_3/split_2/split_dim:output:0'model/up_sampling3d_3/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splite
#model/up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/up_sampling3d_3/concat_2ConcatV2&model/up_sampling3d_3/split_2:output:0&model/up_sampling3d_3/split_2:output:0&model/up_sampling3d_3/split_2:output:1&model/up_sampling3d_3/split_2:output:1&model/up_sampling3d_3/split_2:output:2&model/up_sampling3d_3/split_2:output:2&model/up_sampling3d_3/split_2:output:3&model/up_sampling3d_3/split_2:output:3&model/up_sampling3d_3/split_2:output:4&model/up_sampling3d_3/split_2:output:4&model/up_sampling3d_3/split_2:output:5&model/up_sampling3d_3/split_2:output:5&model/up_sampling3d_3/split_2:output:6&model/up_sampling3d_3/split_2:output:6&model/up_sampling3d_3/split_2:output:7&model/up_sampling3d_3/split_2:output:7&model/up_sampling3d_3/split_2:output:8&model/up_sampling3d_3/split_2:output:8&model/up_sampling3d_3/split_2:output:9&model/up_sampling3d_3/split_2:output:9'model/up_sampling3d_3/split_2:output:10'model/up_sampling3d_3/split_2:output:10'model/up_sampling3d_3/split_2:output:11'model/up_sampling3d_3/split_2:output:11'model/up_sampling3d_3/split_2:output:12'model/up_sampling3d_3/split_2:output:12'model/up_sampling3d_3/split_2:output:13'model/up_sampling3d_3/split_2:output:13'model/up_sampling3d_3/split_2:output:14'model/up_sampling3d_3/split_2:output:14'model/up_sampling3d_3/split_2:output:15'model/up_sampling3d_3/split_2:output:15,model/up_sampling3d_3/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   g
%model/up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_1/splitSplit.model/up_sampling3d_1/split/split_dim:output:0"model/conv3d_17/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч

model/up_sampling3d_1/concatConcatV2$model/up_sampling3d_1/split:output:0$model/up_sampling3d_1/split:output:0$model/up_sampling3d_1/split:output:1$model/up_sampling3d_1/split:output:1$model/up_sampling3d_1/split:output:2$model/up_sampling3d_1/split:output:2$model/up_sampling3d_1/split:output:3$model/up_sampling3d_1/split:output:3$model/up_sampling3d_1/split:output:4$model/up_sampling3d_1/split:output:4$model/up_sampling3d_1/split:output:5$model/up_sampling3d_1/split:output:5$model/up_sampling3d_1/split:output:6$model/up_sampling3d_1/split:output:6$model/up_sampling3d_1/split:output:7$model/up_sampling3d_1/split:output:7$model/up_sampling3d_1/split:output:8$model/up_sampling3d_1/split:output:8$model/up_sampling3d_1/split:output:9$model/up_sampling3d_1/split:output:9%model/up_sampling3d_1/split:output:10%model/up_sampling3d_1/split:output:10%model/up_sampling3d_1/split:output:11%model/up_sampling3d_1/split:output:11%model/up_sampling3d_1/split:output:12%model/up_sampling3d_1/split:output:12%model/up_sampling3d_1/split:output:13%model/up_sampling3d_1/split:output:13%model/up_sampling3d_1/split:output:14%model/up_sampling3d_1/split:output:14%model/up_sampling3d_1/split:output:15%model/up_sampling3d_1/split:output:15*model/up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ i
'model/up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
model/up_sampling3d_1/split_1Split0model/up_sampling3d_1/split_1/split_dim:output:0%model/up_sampling3d_1/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splite
#model/up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/up_sampling3d_1/concat_1ConcatV2&model/up_sampling3d_1/split_1:output:0&model/up_sampling3d_1/split_1:output:0&model/up_sampling3d_1/split_1:output:1&model/up_sampling3d_1/split_1:output:1&model/up_sampling3d_1/split_1:output:2&model/up_sampling3d_1/split_1:output:2&model/up_sampling3d_1/split_1:output:3&model/up_sampling3d_1/split_1:output:3&model/up_sampling3d_1/split_1:output:4&model/up_sampling3d_1/split_1:output:4&model/up_sampling3d_1/split_1:output:5&model/up_sampling3d_1/split_1:output:5&model/up_sampling3d_1/split_1:output:6&model/up_sampling3d_1/split_1:output:6&model/up_sampling3d_1/split_1:output:7&model/up_sampling3d_1/split_1:output:7&model/up_sampling3d_1/split_1:output:8&model/up_sampling3d_1/split_1:output:8&model/up_sampling3d_1/split_1:output:9&model/up_sampling3d_1/split_1:output:9'model/up_sampling3d_1/split_1:output:10'model/up_sampling3d_1/split_1:output:10'model/up_sampling3d_1/split_1:output:11'model/up_sampling3d_1/split_1:output:11'model/up_sampling3d_1/split_1:output:12'model/up_sampling3d_1/split_1:output:12'model/up_sampling3d_1/split_1:output:13'model/up_sampling3d_1/split_1:output:13'model/up_sampling3d_1/split_1:output:14'model/up_sampling3d_1/split_1:output:14'model/up_sampling3d_1/split_1:output:15'model/up_sampling3d_1/split_1:output:15,model/up_sampling3d_1/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  i
'model/up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
model/up_sampling3d_1/split_2Split0model/up_sampling3d_1/split_2/split_dim:output:0'model/up_sampling3d_1/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splite
#model/up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/up_sampling3d_1/concat_2ConcatV2&model/up_sampling3d_1/split_2:output:0&model/up_sampling3d_1/split_2:output:0&model/up_sampling3d_1/split_2:output:1&model/up_sampling3d_1/split_2:output:1&model/up_sampling3d_1/split_2:output:2&model/up_sampling3d_1/split_2:output:2&model/up_sampling3d_1/split_2:output:3&model/up_sampling3d_1/split_2:output:3&model/up_sampling3d_1/split_2:output:4&model/up_sampling3d_1/split_2:output:4&model/up_sampling3d_1/split_2:output:5&model/up_sampling3d_1/split_2:output:5&model/up_sampling3d_1/split_2:output:6&model/up_sampling3d_1/split_2:output:6&model/up_sampling3d_1/split_2:output:7&model/up_sampling3d_1/split_2:output:7&model/up_sampling3d_1/split_2:output:8&model/up_sampling3d_1/split_2:output:8&model/up_sampling3d_1/split_2:output:9&model/up_sampling3d_1/split_2:output:9'model/up_sampling3d_1/split_2:output:10'model/up_sampling3d_1/split_2:output:10'model/up_sampling3d_1/split_2:output:11'model/up_sampling3d_1/split_2:output:11'model/up_sampling3d_1/split_2:output:12'model/up_sampling3d_1/split_2:output:12'model/up_sampling3d_1/split_2:output:13'model/up_sampling3d_1/split_2:output:13'model/up_sampling3d_1/split_2:output:14'model/up_sampling3d_1/split_2:output:14'model/up_sampling3d_1/split_2:output:15'model/up_sampling3d_1/split_2:output:15,model/up_sampling3d_1/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_28/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_28_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0п
model/conv3d_28/Conv3DConv3D'model/up_sampling3d_5/concat_2:output:0-model/conv3d_28/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_28/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_28/BiasAddBiasAddmodel/conv3d_28/Conv3D:output:0.model/conv3d_28/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_28/ReluRelu model/conv3d_28/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_23/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_23_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0п
model/conv3d_23/Conv3DConv3D'model/up_sampling3d_3/concat_2:output:0-model/conv3d_23/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_23/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_23/BiasAddBiasAddmodel/conv3d_23/Conv3D:output:0.model/conv3d_23/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_23/ReluRelu model/conv3d_23/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_18/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0п
model/conv3d_18/Conv3DConv3D'model/up_sampling3d_1/concat_2:output:0-model/conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_18/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_18/BiasAddBiasAddmodel/conv3d_18/Conv3D:output:0.model/conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_18/ReluRelu model/conv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_19/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_19/Conv3DConv3D"model/conv3d_18/Relu:activations:0-model/conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_19/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_19/BiasAddBiasAddmodel/conv3d_19/Conv3D:output:0.model/conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_19/ReluRelu model/conv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_24/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_24_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_24/Conv3DConv3D"model/conv3d_23/Relu:activations:0-model/conv3d_24/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_24/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_24/BiasAddBiasAddmodel/conv3d_24/Conv3D:output:0.model/conv3d_24/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_24/ReluRelu model/conv3d_24/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_29/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_29_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_29/Conv3DConv3D"model/conv3d_28/Relu:activations:0-model/conv3d_29/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_29/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_29/BiasAddBiasAddmodel/conv3d_29/Conv3D:output:0.model/conv3d_29/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   |
model/conv3d_29/ReluRelu model/conv3d_29/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
model/add_1/addAddV2"model/conv3d_19/Relu:activations:0"model/conv3d_24/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
model/add_1/add_1AddV2model/add_1/add:z:0"model/conv3d_29/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ    
%model/conv3d_30/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_30_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
model/conv3d_30/Conv3DConv3Dmodel/add_1/add_1:z:0-model/conv3d_30/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

&model/conv3d_30/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_30/BiasAddBiasAddmodel/conv3d_30/Conv3D:output:0.model/conv3d_30/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   {
IdentityIdentity model/conv3d_30/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   §
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp'^model/conv3d_10/BiasAdd/ReadVariableOp&^model/conv3d_10/Conv3D/ReadVariableOp'^model/conv3d_11/BiasAdd/ReadVariableOp&^model/conv3d_11/Conv3D/ReadVariableOp'^model/conv3d_12/BiasAdd/ReadVariableOp&^model/conv3d_12/Conv3D/ReadVariableOp'^model/conv3d_13/BiasAdd/ReadVariableOp&^model/conv3d_13/Conv3D/ReadVariableOp'^model/conv3d_14/BiasAdd/ReadVariableOp&^model/conv3d_14/Conv3D/ReadVariableOp'^model/conv3d_15/BiasAdd/ReadVariableOp&^model/conv3d_15/Conv3D/ReadVariableOp'^model/conv3d_16/BiasAdd/ReadVariableOp&^model/conv3d_16/Conv3D/ReadVariableOp'^model/conv3d_17/BiasAdd/ReadVariableOp&^model/conv3d_17/Conv3D/ReadVariableOp'^model/conv3d_18/BiasAdd/ReadVariableOp&^model/conv3d_18/Conv3D/ReadVariableOp'^model/conv3d_19/BiasAdd/ReadVariableOp&^model/conv3d_19/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp'^model/conv3d_20/BiasAdd/ReadVariableOp&^model/conv3d_20/Conv3D/ReadVariableOp'^model/conv3d_21/BiasAdd/ReadVariableOp&^model/conv3d_21/Conv3D/ReadVariableOp'^model/conv3d_22/BiasAdd/ReadVariableOp&^model/conv3d_22/Conv3D/ReadVariableOp'^model/conv3d_23/BiasAdd/ReadVariableOp&^model/conv3d_23/Conv3D/ReadVariableOp'^model/conv3d_24/BiasAdd/ReadVariableOp&^model/conv3d_24/Conv3D/ReadVariableOp'^model/conv3d_25/BiasAdd/ReadVariableOp&^model/conv3d_25/Conv3D/ReadVariableOp'^model/conv3d_26/BiasAdd/ReadVariableOp&^model/conv3d_26/Conv3D/ReadVariableOp'^model/conv3d_27/BiasAdd/ReadVariableOp&^model/conv3d_27/Conv3D/ReadVariableOp'^model/conv3d_28/BiasAdd/ReadVariableOp&^model/conv3d_28/Conv3D/ReadVariableOp'^model/conv3d_29/BiasAdd/ReadVariableOp&^model/conv3d_29/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp'^model/conv3d_30/BiasAdd/ReadVariableOp&^model/conv3d_30/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp&^model/conv3d_8/BiasAdd/ReadVariableOp%^model/conv3d_8/Conv3D/ReadVariableOp&^model/conv3d_9/BiasAdd/ReadVariableOp%^model/conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2P
&model/conv3d_10/BiasAdd/ReadVariableOp&model/conv3d_10/BiasAdd/ReadVariableOp2N
%model/conv3d_10/Conv3D/ReadVariableOp%model/conv3d_10/Conv3D/ReadVariableOp2P
&model/conv3d_11/BiasAdd/ReadVariableOp&model/conv3d_11/BiasAdd/ReadVariableOp2N
%model/conv3d_11/Conv3D/ReadVariableOp%model/conv3d_11/Conv3D/ReadVariableOp2P
&model/conv3d_12/BiasAdd/ReadVariableOp&model/conv3d_12/BiasAdd/ReadVariableOp2N
%model/conv3d_12/Conv3D/ReadVariableOp%model/conv3d_12/Conv3D/ReadVariableOp2P
&model/conv3d_13/BiasAdd/ReadVariableOp&model/conv3d_13/BiasAdd/ReadVariableOp2N
%model/conv3d_13/Conv3D/ReadVariableOp%model/conv3d_13/Conv3D/ReadVariableOp2P
&model/conv3d_14/BiasAdd/ReadVariableOp&model/conv3d_14/BiasAdd/ReadVariableOp2N
%model/conv3d_14/Conv3D/ReadVariableOp%model/conv3d_14/Conv3D/ReadVariableOp2P
&model/conv3d_15/BiasAdd/ReadVariableOp&model/conv3d_15/BiasAdd/ReadVariableOp2N
%model/conv3d_15/Conv3D/ReadVariableOp%model/conv3d_15/Conv3D/ReadVariableOp2P
&model/conv3d_16/BiasAdd/ReadVariableOp&model/conv3d_16/BiasAdd/ReadVariableOp2N
%model/conv3d_16/Conv3D/ReadVariableOp%model/conv3d_16/Conv3D/ReadVariableOp2P
&model/conv3d_17/BiasAdd/ReadVariableOp&model/conv3d_17/BiasAdd/ReadVariableOp2N
%model/conv3d_17/Conv3D/ReadVariableOp%model/conv3d_17/Conv3D/ReadVariableOp2P
&model/conv3d_18/BiasAdd/ReadVariableOp&model/conv3d_18/BiasAdd/ReadVariableOp2N
%model/conv3d_18/Conv3D/ReadVariableOp%model/conv3d_18/Conv3D/ReadVariableOp2P
&model/conv3d_19/BiasAdd/ReadVariableOp&model/conv3d_19/BiasAdd/ReadVariableOp2N
%model/conv3d_19/Conv3D/ReadVariableOp%model/conv3d_19/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2P
&model/conv3d_20/BiasAdd/ReadVariableOp&model/conv3d_20/BiasAdd/ReadVariableOp2N
%model/conv3d_20/Conv3D/ReadVariableOp%model/conv3d_20/Conv3D/ReadVariableOp2P
&model/conv3d_21/BiasAdd/ReadVariableOp&model/conv3d_21/BiasAdd/ReadVariableOp2N
%model/conv3d_21/Conv3D/ReadVariableOp%model/conv3d_21/Conv3D/ReadVariableOp2P
&model/conv3d_22/BiasAdd/ReadVariableOp&model/conv3d_22/BiasAdd/ReadVariableOp2N
%model/conv3d_22/Conv3D/ReadVariableOp%model/conv3d_22/Conv3D/ReadVariableOp2P
&model/conv3d_23/BiasAdd/ReadVariableOp&model/conv3d_23/BiasAdd/ReadVariableOp2N
%model/conv3d_23/Conv3D/ReadVariableOp%model/conv3d_23/Conv3D/ReadVariableOp2P
&model/conv3d_24/BiasAdd/ReadVariableOp&model/conv3d_24/BiasAdd/ReadVariableOp2N
%model/conv3d_24/Conv3D/ReadVariableOp%model/conv3d_24/Conv3D/ReadVariableOp2P
&model/conv3d_25/BiasAdd/ReadVariableOp&model/conv3d_25/BiasAdd/ReadVariableOp2N
%model/conv3d_25/Conv3D/ReadVariableOp%model/conv3d_25/Conv3D/ReadVariableOp2P
&model/conv3d_26/BiasAdd/ReadVariableOp&model/conv3d_26/BiasAdd/ReadVariableOp2N
%model/conv3d_26/Conv3D/ReadVariableOp%model/conv3d_26/Conv3D/ReadVariableOp2P
&model/conv3d_27/BiasAdd/ReadVariableOp&model/conv3d_27/BiasAdd/ReadVariableOp2N
%model/conv3d_27/Conv3D/ReadVariableOp%model/conv3d_27/Conv3D/ReadVariableOp2P
&model/conv3d_28/BiasAdd/ReadVariableOp&model/conv3d_28/BiasAdd/ReadVariableOp2N
%model/conv3d_28/Conv3D/ReadVariableOp%model/conv3d_28/Conv3D/ReadVariableOp2P
&model/conv3d_29/BiasAdd/ReadVariableOp&model/conv3d_29/BiasAdd/ReadVariableOp2N
%model/conv3d_29/Conv3D/ReadVariableOp%model/conv3d_29/Conv3D/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2P
&model/conv3d_30/BiasAdd/ReadVariableOp&model/conv3d_30/BiasAdd/ReadVariableOp2N
%model/conv3d_30/Conv3D/ReadVariableOp%model/conv3d_30/Conv3D/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp2N
%model/conv3d_6/BiasAdd/ReadVariableOp%model/conv3d_6/BiasAdd/ReadVariableOp2L
$model/conv3d_6/Conv3D/ReadVariableOp$model/conv3d_6/Conv3D/ReadVariableOp2N
%model/conv3d_7/BiasAdd/ReadVariableOp%model/conv3d_7/BiasAdd/ReadVariableOp2L
$model/conv3d_7/Conv3D/ReadVariableOp$model/conv3d_7/Conv3D/ReadVariableOp2N
%model/conv3d_8/BiasAdd/ReadVariableOp%model/conv3d_8/BiasAdd/ReadVariableOp2L
$model/conv3d_8/Conv3D/ReadVariableOp$model/conv3d_8/Conv3D/ReadVariableOp2N
%model/conv3d_9/BiasAdd/ReadVariableOp%model/conv3d_9/BiasAdd/ReadVariableOp2L
$model/conv3d_9/Conv3D/ReadVariableOp$model/conv3d_9/Conv3D/ReadVariableOp:V R
/
_output_shapes
:џџџџџџџџџ   

_user_specified_nameu_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namev_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namew_vel
ѕ
{
A__inference_add_1_layer_call_and_return_conditional_losses_136712
inputs_0
inputs_1
inputs_2
identity^
addAddV2inputs_0inputs_1*
T0*3
_output_shapes!
:џџџџџџџџџ   _
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:џџџџџџџџџ   ]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   :] Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/2


D__inference_conv3d_9_layer_call_and_return_conditional_losses_136001

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџc
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_22_layer_call_fn_136348

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_22_layer_call_and_return_conditional_losses_132179{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Я
_
C__inference_reshape_layer_call_and_return_conditional_losses_131554

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
б
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_131571

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_22_layer_call_and_return_conditional_losses_136359

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
У
a
E__inference_reshape_7_layer_call_and_return_conditional_losses_131930

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџd
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ
y
?__inference_add_layer_call_and_return_conditional_losses_136036
inputs_0
inputs_1
inputs_2
identity^
addAddV2inputs_0inputs_1*
T0*3
_output_shapes!
:џџџџџџџџџ_
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:џџџџџџџџџ]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:] Y
3
_output_shapes!
:џџџџџџџџџ
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:џџџџџџџџџ
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Ї

E__inference_conv3d_29_layer_call_and_return_conditional_losses_136697

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
н
Ё
"__inference__traced_restore_137924
file_prefix<
assignvariableop_conv3d_kernel:,
assignvariableop_1_conv3d_bias:@
"assignvariableop_2_conv3d_5_kernel:.
 assignvariableop_3_conv3d_5_bias:A
#assignvariableop_4_conv3d_10_kernel:/
!assignvariableop_5_conv3d_10_bias:@
"assignvariableop_6_conv3d_1_kernel:.
 assignvariableop_7_conv3d_1_bias:@
"assignvariableop_8_conv3d_6_kernel:.
 assignvariableop_9_conv3d_6_bias:B
$assignvariableop_10_conv3d_11_kernel:0
"assignvariableop_11_conv3d_11_bias:A
#assignvariableop_12_conv3d_2_kernel:/
!assignvariableop_13_conv3d_2_bias:A
#assignvariableop_14_conv3d_7_kernel:/
!assignvariableop_15_conv3d_7_bias:B
$assignvariableop_16_conv3d_12_kernel:0
"assignvariableop_17_conv3d_12_bias:A
#assignvariableop_18_conv3d_3_kernel:/
!assignvariableop_19_conv3d_3_bias:A
#assignvariableop_20_conv3d_8_kernel:/
!assignvariableop_21_conv3d_8_bias:B
$assignvariableop_22_conv3d_13_kernel:0
"assignvariableop_23_conv3d_13_bias:A
#assignvariableop_24_conv3d_4_kernel:/
!assignvariableop_25_conv3d_4_bias:A
#assignvariableop_26_conv3d_9_kernel:/
!assignvariableop_27_conv3d_9_bias:B
$assignvariableop_28_conv3d_14_kernel:0
"assignvariableop_29_conv3d_14_bias:B
$assignvariableop_30_conv3d_15_kernel:0
"assignvariableop_31_conv3d_15_bias:B
$assignvariableop_32_conv3d_20_kernel:0
"assignvariableop_33_conv3d_20_bias:B
$assignvariableop_34_conv3d_25_kernel:0
"assignvariableop_35_conv3d_25_bias:B
$assignvariableop_36_conv3d_16_kernel:0
"assignvariableop_37_conv3d_16_bias:B
$assignvariableop_38_conv3d_21_kernel:0
"assignvariableop_39_conv3d_21_bias:B
$assignvariableop_40_conv3d_26_kernel:0
"assignvariableop_41_conv3d_26_bias:B
$assignvariableop_42_conv3d_17_kernel:0
"assignvariableop_43_conv3d_17_bias:B
$assignvariableop_44_conv3d_22_kernel:0
"assignvariableop_45_conv3d_22_bias:B
$assignvariableop_46_conv3d_27_kernel:0
"assignvariableop_47_conv3d_27_bias:B
$assignvariableop_48_conv3d_18_kernel:0
"assignvariableop_49_conv3d_18_bias:B
$assignvariableop_50_conv3d_23_kernel:0
"assignvariableop_51_conv3d_23_bias:B
$assignvariableop_52_conv3d_28_kernel:0
"assignvariableop_53_conv3d_28_bias:B
$assignvariableop_54_conv3d_19_kernel:0
"assignvariableop_55_conv3d_19_bias:B
$assignvariableop_56_conv3d_24_kernel:0
"assignvariableop_57_conv3d_24_bias:B
$assignvariableop_58_conv3d_29_kernel:0
"assignvariableop_59_conv3d_29_bias:B
$assignvariableop_60_conv3d_30_kernel:0
"assignvariableop_61_conv3d_30_bias:'
assignvariableop_62_adam_iter:	 )
assignvariableop_63_adam_beta_1: )
assignvariableop_64_adam_beta_2: (
assignvariableop_65_adam_decay: 0
&assignvariableop_66_adam_learning_rate: #
assignvariableop_67_total: #
assignvariableop_68_count: F
(assignvariableop_69_adam_conv3d_kernel_m:4
&assignvariableop_70_adam_conv3d_bias_m:H
*assignvariableop_71_adam_conv3d_5_kernel_m:6
(assignvariableop_72_adam_conv3d_5_bias_m:I
+assignvariableop_73_adam_conv3d_10_kernel_m:7
)assignvariableop_74_adam_conv3d_10_bias_m:H
*assignvariableop_75_adam_conv3d_1_kernel_m:6
(assignvariableop_76_adam_conv3d_1_bias_m:H
*assignvariableop_77_adam_conv3d_6_kernel_m:6
(assignvariableop_78_adam_conv3d_6_bias_m:I
+assignvariableop_79_adam_conv3d_11_kernel_m:7
)assignvariableop_80_adam_conv3d_11_bias_m:H
*assignvariableop_81_adam_conv3d_2_kernel_m:6
(assignvariableop_82_adam_conv3d_2_bias_m:H
*assignvariableop_83_adam_conv3d_7_kernel_m:6
(assignvariableop_84_adam_conv3d_7_bias_m:I
+assignvariableop_85_adam_conv3d_12_kernel_m:7
)assignvariableop_86_adam_conv3d_12_bias_m:H
*assignvariableop_87_adam_conv3d_3_kernel_m:6
(assignvariableop_88_adam_conv3d_3_bias_m:H
*assignvariableop_89_adam_conv3d_8_kernel_m:6
(assignvariableop_90_adam_conv3d_8_bias_m:I
+assignvariableop_91_adam_conv3d_13_kernel_m:7
)assignvariableop_92_adam_conv3d_13_bias_m:H
*assignvariableop_93_adam_conv3d_4_kernel_m:6
(assignvariableop_94_adam_conv3d_4_bias_m:H
*assignvariableop_95_adam_conv3d_9_kernel_m:6
(assignvariableop_96_adam_conv3d_9_bias_m:I
+assignvariableop_97_adam_conv3d_14_kernel_m:7
)assignvariableop_98_adam_conv3d_14_bias_m:I
+assignvariableop_99_adam_conv3d_15_kernel_m:8
*assignvariableop_100_adam_conv3d_15_bias_m:J
,assignvariableop_101_adam_conv3d_20_kernel_m:8
*assignvariableop_102_adam_conv3d_20_bias_m:J
,assignvariableop_103_adam_conv3d_25_kernel_m:8
*assignvariableop_104_adam_conv3d_25_bias_m:J
,assignvariableop_105_adam_conv3d_16_kernel_m:8
*assignvariableop_106_adam_conv3d_16_bias_m:J
,assignvariableop_107_adam_conv3d_21_kernel_m:8
*assignvariableop_108_adam_conv3d_21_bias_m:J
,assignvariableop_109_adam_conv3d_26_kernel_m:8
*assignvariableop_110_adam_conv3d_26_bias_m:J
,assignvariableop_111_adam_conv3d_17_kernel_m:8
*assignvariableop_112_adam_conv3d_17_bias_m:J
,assignvariableop_113_adam_conv3d_22_kernel_m:8
*assignvariableop_114_adam_conv3d_22_bias_m:J
,assignvariableop_115_adam_conv3d_27_kernel_m:8
*assignvariableop_116_adam_conv3d_27_bias_m:J
,assignvariableop_117_adam_conv3d_18_kernel_m:8
*assignvariableop_118_adam_conv3d_18_bias_m:J
,assignvariableop_119_adam_conv3d_23_kernel_m:8
*assignvariableop_120_adam_conv3d_23_bias_m:J
,assignvariableop_121_adam_conv3d_28_kernel_m:8
*assignvariableop_122_adam_conv3d_28_bias_m:J
,assignvariableop_123_adam_conv3d_19_kernel_m:8
*assignvariableop_124_adam_conv3d_19_bias_m:J
,assignvariableop_125_adam_conv3d_24_kernel_m:8
*assignvariableop_126_adam_conv3d_24_bias_m:J
,assignvariableop_127_adam_conv3d_29_kernel_m:8
*assignvariableop_128_adam_conv3d_29_bias_m:J
,assignvariableop_129_adam_conv3d_30_kernel_m:8
*assignvariableop_130_adam_conv3d_30_bias_m:G
)assignvariableop_131_adam_conv3d_kernel_v:5
'assignvariableop_132_adam_conv3d_bias_v:I
+assignvariableop_133_adam_conv3d_5_kernel_v:7
)assignvariableop_134_adam_conv3d_5_bias_v:J
,assignvariableop_135_adam_conv3d_10_kernel_v:8
*assignvariableop_136_adam_conv3d_10_bias_v:I
+assignvariableop_137_adam_conv3d_1_kernel_v:7
)assignvariableop_138_adam_conv3d_1_bias_v:I
+assignvariableop_139_adam_conv3d_6_kernel_v:7
)assignvariableop_140_adam_conv3d_6_bias_v:J
,assignvariableop_141_adam_conv3d_11_kernel_v:8
*assignvariableop_142_adam_conv3d_11_bias_v:I
+assignvariableop_143_adam_conv3d_2_kernel_v:7
)assignvariableop_144_adam_conv3d_2_bias_v:I
+assignvariableop_145_adam_conv3d_7_kernel_v:7
)assignvariableop_146_adam_conv3d_7_bias_v:J
,assignvariableop_147_adam_conv3d_12_kernel_v:8
*assignvariableop_148_adam_conv3d_12_bias_v:I
+assignvariableop_149_adam_conv3d_3_kernel_v:7
)assignvariableop_150_adam_conv3d_3_bias_v:I
+assignvariableop_151_adam_conv3d_8_kernel_v:7
)assignvariableop_152_adam_conv3d_8_bias_v:J
,assignvariableop_153_adam_conv3d_13_kernel_v:8
*assignvariableop_154_adam_conv3d_13_bias_v:I
+assignvariableop_155_adam_conv3d_4_kernel_v:7
)assignvariableop_156_adam_conv3d_4_bias_v:I
+assignvariableop_157_adam_conv3d_9_kernel_v:7
)assignvariableop_158_adam_conv3d_9_bias_v:J
,assignvariableop_159_adam_conv3d_14_kernel_v:8
*assignvariableop_160_adam_conv3d_14_bias_v:J
,assignvariableop_161_adam_conv3d_15_kernel_v:8
*assignvariableop_162_adam_conv3d_15_bias_v:J
,assignvariableop_163_adam_conv3d_20_kernel_v:8
*assignvariableop_164_adam_conv3d_20_bias_v:J
,assignvariableop_165_adam_conv3d_25_kernel_v:8
*assignvariableop_166_adam_conv3d_25_bias_v:J
,assignvariableop_167_adam_conv3d_16_kernel_v:8
*assignvariableop_168_adam_conv3d_16_bias_v:J
,assignvariableop_169_adam_conv3d_21_kernel_v:8
*assignvariableop_170_adam_conv3d_21_bias_v:J
,assignvariableop_171_adam_conv3d_26_kernel_v:8
*assignvariableop_172_adam_conv3d_26_bias_v:J
,assignvariableop_173_adam_conv3d_17_kernel_v:8
*assignvariableop_174_adam_conv3d_17_bias_v:J
,assignvariableop_175_adam_conv3d_22_kernel_v:8
*assignvariableop_176_adam_conv3d_22_bias_v:J
,assignvariableop_177_adam_conv3d_27_kernel_v:8
*assignvariableop_178_adam_conv3d_27_bias_v:J
,assignvariableop_179_adam_conv3d_18_kernel_v:8
*assignvariableop_180_adam_conv3d_18_bias_v:J
,assignvariableop_181_adam_conv3d_23_kernel_v:8
*assignvariableop_182_adam_conv3d_23_bias_v:J
,assignvariableop_183_adam_conv3d_28_kernel_v:8
*assignvariableop_184_adam_conv3d_28_bias_v:J
,assignvariableop_185_adam_conv3d_19_kernel_v:8
*assignvariableop_186_adam_conv3d_19_bias_v:J
,assignvariableop_187_adam_conv3d_24_kernel_v:8
*assignvariableop_188_adam_conv3d_24_bias_v:J
,assignvariableop_189_adam_conv3d_29_kernel_v:8
*assignvariableop_190_adam_conv3d_29_bias_v:J
,assignvariableop_191_adam_conv3d_30_kernel_v:8
*assignvariableop_192_adam_conv3d_30_bias_v:
identity_194ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_101ЂAssignVariableOp_102ЂAssignVariableOp_103ЂAssignVariableOp_104ЂAssignVariableOp_105ЂAssignVariableOp_106ЂAssignVariableOp_107ЂAssignVariableOp_108ЂAssignVariableOp_109ЂAssignVariableOp_11ЂAssignVariableOp_110ЂAssignVariableOp_111ЂAssignVariableOp_112ЂAssignVariableOp_113ЂAssignVariableOp_114ЂAssignVariableOp_115ЂAssignVariableOp_116ЂAssignVariableOp_117ЂAssignVariableOp_118ЂAssignVariableOp_119ЂAssignVariableOp_12ЂAssignVariableOp_120ЂAssignVariableOp_121ЂAssignVariableOp_122ЂAssignVariableOp_123ЂAssignVariableOp_124ЂAssignVariableOp_125ЂAssignVariableOp_126ЂAssignVariableOp_127ЂAssignVariableOp_128ЂAssignVariableOp_129ЂAssignVariableOp_13ЂAssignVariableOp_130ЂAssignVariableOp_131ЂAssignVariableOp_132ЂAssignVariableOp_133ЂAssignVariableOp_134ЂAssignVariableOp_135ЂAssignVariableOp_136ЂAssignVariableOp_137ЂAssignVariableOp_138ЂAssignVariableOp_139ЂAssignVariableOp_14ЂAssignVariableOp_140ЂAssignVariableOp_141ЂAssignVariableOp_142ЂAssignVariableOp_143ЂAssignVariableOp_144ЂAssignVariableOp_145ЂAssignVariableOp_146ЂAssignVariableOp_147ЂAssignVariableOp_148ЂAssignVariableOp_149ЂAssignVariableOp_15ЂAssignVariableOp_150ЂAssignVariableOp_151ЂAssignVariableOp_152ЂAssignVariableOp_153ЂAssignVariableOp_154ЂAssignVariableOp_155ЂAssignVariableOp_156ЂAssignVariableOp_157ЂAssignVariableOp_158ЂAssignVariableOp_159ЂAssignVariableOp_16ЂAssignVariableOp_160ЂAssignVariableOp_161ЂAssignVariableOp_162ЂAssignVariableOp_163ЂAssignVariableOp_164ЂAssignVariableOp_165ЂAssignVariableOp_166ЂAssignVariableOp_167ЂAssignVariableOp_168ЂAssignVariableOp_169ЂAssignVariableOp_17ЂAssignVariableOp_170ЂAssignVariableOp_171ЂAssignVariableOp_172ЂAssignVariableOp_173ЂAssignVariableOp_174ЂAssignVariableOp_175ЂAssignVariableOp_176ЂAssignVariableOp_177ЂAssignVariableOp_178ЂAssignVariableOp_179ЂAssignVariableOp_18ЂAssignVariableOp_180ЂAssignVariableOp_181ЂAssignVariableOp_182ЂAssignVariableOp_183ЂAssignVariableOp_184ЂAssignVariableOp_185ЂAssignVariableOp_186ЂAssignVariableOp_187ЂAssignVariableOp_188ЂAssignVariableOp_189ЂAssignVariableOp_19ЂAssignVariableOp_190ЂAssignVariableOp_191ЂAssignVariableOp_192ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_99фo
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:Т*
dtype0*o
valueџnBќnТB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHљ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:Т*
dtype0*
valueBТB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B я
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*г
dtypesШ
Х2Т	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv3d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv3d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv3d_11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv3d_11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv3d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv3d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv3d_12_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv3d_12_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv3d_8_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv3d_8_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv3d_13_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv3d_13_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv3d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv3d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv3d_9_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp!assignvariableop_27_conv3d_9_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv3d_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv3d_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv3d_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv3d_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv3d_20_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv3d_20_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv3d_25_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv3d_25_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv3d_16_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv3d_16_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv3d_21_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv3d_21_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv3d_26_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv3d_26_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv3d_17_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv3d_17_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp$assignvariableop_44_conv3d_22_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv3d_22_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp$assignvariableop_46_conv3d_27_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp"assignvariableop_47_conv3d_27_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv3d_18_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv3d_18_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp$assignvariableop_50_conv3d_23_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp"assignvariableop_51_conv3d_23_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp$assignvariableop_52_conv3d_28_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp"assignvariableop_53_conv3d_28_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp$assignvariableop_54_conv3d_19_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp"assignvariableop_55_conv3d_19_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp$assignvariableop_56_conv3d_24_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp"assignvariableop_57_conv3d_24_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp$assignvariableop_58_conv3d_29_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp"assignvariableop_59_conv3d_29_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp$assignvariableop_60_conv3d_30_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp"assignvariableop_61_conv3d_30_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_62AssignVariableOpassignvariableop_62_adam_iterIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOpassignvariableop_63_adam_beta_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOpassignvariableop_64_adam_beta_2Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_decayIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp&assignvariableop_66_adam_learning_rateIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOpassignvariableop_67_totalIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOpassignvariableop_68_countIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_conv3d_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_conv3d_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv3d_5_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv3d_5_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv3d_10_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv3d_10_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv3d_1_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv3d_1_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv3d_6_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv3d_6_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv3d_11_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv3d_11_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv3d_2_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv3d_2_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv3d_7_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv3d_7_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv3d_12_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv3d_12_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv3d_3_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv3d_3_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv3d_8_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv3d_8_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv3d_13_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv3d_13_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_conv3d_4_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_conv3d_4_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_conv3d_9_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_conv3d_9_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv3d_14_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv3d_14_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_conv3d_15_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_conv3d_15_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv3d_20_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv3d_20_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv3d_25_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv3d_25_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv3d_16_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv3d_16_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv3d_21_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv3d_21_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv3d_26_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv3d_26_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_conv3d_17_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_conv3d_17_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv3d_22_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv3d_22_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_conv3d_27_kernel_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_conv3d_27_bias_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv3d_18_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv3d_18_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_conv3d_23_kernel_mIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_conv3d_23_bias_mIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv3d_28_kernel_mIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv3d_28_bias_mIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_conv3d_19_kernel_mIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_conv3d_19_bias_mIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_conv3d_24_kernel_mIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_conv3d_24_bias_mIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_conv3d_29_kernel_mIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_conv3d_29_bias_mIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_129AssignVariableOp,assignvariableop_129_adam_conv3d_30_kernel_mIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_130AssignVariableOp*assignvariableop_130_adam_conv3d_30_bias_mIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp)assignvariableop_131_adam_conv3d_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp'assignvariableop_132_adam_conv3d_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_133AssignVariableOp+assignvariableop_133_adam_conv3d_5_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_134AssignVariableOp)assignvariableop_134_adam_conv3d_5_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_conv3d_10_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_conv3d_10_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_137AssignVariableOp+assignvariableop_137_adam_conv3d_1_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_138AssignVariableOp)assignvariableop_138_adam_conv3d_1_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_139AssignVariableOp+assignvariableop_139_adam_conv3d_6_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_140AssignVariableOp)assignvariableop_140_adam_conv3d_6_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_141AssignVariableOp,assignvariableop_141_adam_conv3d_11_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_142AssignVariableOp*assignvariableop_142_adam_conv3d_11_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_143AssignVariableOp+assignvariableop_143_adam_conv3d_2_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_144AssignVariableOp)assignvariableop_144_adam_conv3d_2_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_145AssignVariableOp+assignvariableop_145_adam_conv3d_7_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_146AssignVariableOp)assignvariableop_146_adam_conv3d_7_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_147AssignVariableOp,assignvariableop_147_adam_conv3d_12_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_148AssignVariableOp*assignvariableop_148_adam_conv3d_12_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_149AssignVariableOp+assignvariableop_149_adam_conv3d_3_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_150AssignVariableOp)assignvariableop_150_adam_conv3d_3_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_151AssignVariableOp+assignvariableop_151_adam_conv3d_8_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_152AssignVariableOp)assignvariableop_152_adam_conv3d_8_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_153AssignVariableOp,assignvariableop_153_adam_conv3d_13_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_154AssignVariableOp*assignvariableop_154_adam_conv3d_13_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_155AssignVariableOp+assignvariableop_155_adam_conv3d_4_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_156AssignVariableOp)assignvariableop_156_adam_conv3d_4_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_157AssignVariableOp+assignvariableop_157_adam_conv3d_9_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_158AssignVariableOp)assignvariableop_158_adam_conv3d_9_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_159AssignVariableOp,assignvariableop_159_adam_conv3d_14_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_160AssignVariableOp*assignvariableop_160_adam_conv3d_14_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_161AssignVariableOp,assignvariableop_161_adam_conv3d_15_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_162AssignVariableOp*assignvariableop_162_adam_conv3d_15_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_163AssignVariableOp,assignvariableop_163_adam_conv3d_20_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_164AssignVariableOp*assignvariableop_164_adam_conv3d_20_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_165AssignVariableOp,assignvariableop_165_adam_conv3d_25_kernel_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_166AssignVariableOp*assignvariableop_166_adam_conv3d_25_bias_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_167AssignVariableOp,assignvariableop_167_adam_conv3d_16_kernel_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_168AssignVariableOp*assignvariableop_168_adam_conv3d_16_bias_vIdentity_168:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_169AssignVariableOp,assignvariableop_169_adam_conv3d_21_kernel_vIdentity_169:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_170AssignVariableOp*assignvariableop_170_adam_conv3d_21_bias_vIdentity_170:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_171IdentityRestoreV2:tensors:171"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_171AssignVariableOp,assignvariableop_171_adam_conv3d_26_kernel_vIdentity_171:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_172IdentityRestoreV2:tensors:172"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_172AssignVariableOp*assignvariableop_172_adam_conv3d_26_bias_vIdentity_172:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_173IdentityRestoreV2:tensors:173"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_173AssignVariableOp,assignvariableop_173_adam_conv3d_17_kernel_vIdentity_173:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_174IdentityRestoreV2:tensors:174"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_174AssignVariableOp*assignvariableop_174_adam_conv3d_17_bias_vIdentity_174:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_175IdentityRestoreV2:tensors:175"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_175AssignVariableOp,assignvariableop_175_adam_conv3d_22_kernel_vIdentity_175:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_176IdentityRestoreV2:tensors:176"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_176AssignVariableOp*assignvariableop_176_adam_conv3d_22_bias_vIdentity_176:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_177IdentityRestoreV2:tensors:177"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_177AssignVariableOp,assignvariableop_177_adam_conv3d_27_kernel_vIdentity_177:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_178IdentityRestoreV2:tensors:178"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_178AssignVariableOp*assignvariableop_178_adam_conv3d_27_bias_vIdentity_178:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_179IdentityRestoreV2:tensors:179"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_179AssignVariableOp,assignvariableop_179_adam_conv3d_18_kernel_vIdentity_179:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_180IdentityRestoreV2:tensors:180"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_180AssignVariableOp*assignvariableop_180_adam_conv3d_18_bias_vIdentity_180:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_181IdentityRestoreV2:tensors:181"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_181AssignVariableOp,assignvariableop_181_adam_conv3d_23_kernel_vIdentity_181:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_182IdentityRestoreV2:tensors:182"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_182AssignVariableOp*assignvariableop_182_adam_conv3d_23_bias_vIdentity_182:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_183IdentityRestoreV2:tensors:183"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_183AssignVariableOp,assignvariableop_183_adam_conv3d_28_kernel_vIdentity_183:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_184IdentityRestoreV2:tensors:184"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_184AssignVariableOp*assignvariableop_184_adam_conv3d_28_bias_vIdentity_184:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_185IdentityRestoreV2:tensors:185"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_185AssignVariableOp,assignvariableop_185_adam_conv3d_19_kernel_vIdentity_185:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_186IdentityRestoreV2:tensors:186"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_186AssignVariableOp*assignvariableop_186_adam_conv3d_19_bias_vIdentity_186:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_187IdentityRestoreV2:tensors:187"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_187AssignVariableOp,assignvariableop_187_adam_conv3d_24_kernel_vIdentity_187:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_188IdentityRestoreV2:tensors:188"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_188AssignVariableOp*assignvariableop_188_adam_conv3d_24_bias_vIdentity_188:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_189IdentityRestoreV2:tensors:189"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_189AssignVariableOp,assignvariableop_189_adam_conv3d_29_kernel_vIdentity_189:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_190IdentityRestoreV2:tensors:190"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_190AssignVariableOp*assignvariableop_190_adam_conv3d_29_bias_vIdentity_190:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_191IdentityRestoreV2:tensors:191"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_191AssignVariableOp,assignvariableop_191_adam_conv3d_30_kernel_vIdentity_191:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_192IdentityRestoreV2:tensors:192"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_192AssignVariableOp*assignvariableop_192_adam_conv3d_30_bias_vIdentity_192:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 У"
Identity_193Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_186^AssignVariableOp_187^AssignVariableOp_188^AssignVariableOp_189^AssignVariableOp_19^AssignVariableOp_190^AssignVariableOp_191^AssignVariableOp_192^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_194IdentityIdentity_193:output:0^NoOp_1*
T0*
_output_shapes
: Џ"
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_186^AssignVariableOp_187^AssignVariableOp_188^AssignVariableOp_189^AssignVariableOp_19^AssignVariableOp_190^AssignVariableOp_191^AssignVariableOp_192^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_194Identity_194:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_170AssignVariableOp_1702,
AssignVariableOp_171AssignVariableOp_1712,
AssignVariableOp_172AssignVariableOp_1722,
AssignVariableOp_173AssignVariableOp_1732,
AssignVariableOp_174AssignVariableOp_1742,
AssignVariableOp_175AssignVariableOp_1752,
AssignVariableOp_176AssignVariableOp_1762,
AssignVariableOp_177AssignVariableOp_1772,
AssignVariableOp_178AssignVariableOp_1782,
AssignVariableOp_179AssignVariableOp_1792*
AssignVariableOp_18AssignVariableOp_182,
AssignVariableOp_180AssignVariableOp_1802,
AssignVariableOp_181AssignVariableOp_1812,
AssignVariableOp_182AssignVariableOp_1822,
AssignVariableOp_183AssignVariableOp_1832,
AssignVariableOp_184AssignVariableOp_1842,
AssignVariableOp_185AssignVariableOp_1852,
AssignVariableOp_186AssignVariableOp_1862,
AssignVariableOp_187AssignVariableOp_1872,
AssignVariableOp_188AssignVariableOp_1882,
AssignVariableOp_189AssignVariableOp_1892*
AssignVariableOp_19AssignVariableOp_192,
AssignVariableOp_190AssignVariableOp_1902,
AssignVariableOp_191AssignVariableOp_1912,
AssignVariableOp_192AssignVariableOp_1922(
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
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
І

D__inference_conv3d_6_layer_call_and_return_conditional_losses_131679

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_10_layer_call_fn_135650

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_131611{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
э
J
.__inference_max_pooling3d_layer_call_fn_135726

inputs
identityщ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131465
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
но
н
A__inference_model_layer_call_and_return_conditional_losses_134013	
u_vel	
v_vel	
w_vel.
conv3d_10_133841:
conv3d_10_133843:-
conv3d_5_133846:
conv3d_5_133848:+
conv3d_133851:
conv3d_133853:.
conv3d_11_133856:
conv3d_11_133858:-
conv3d_6_133861:
conv3d_6_133863:-
conv3d_1_133866:
conv3d_1_133868:.
conv3d_12_133874:
conv3d_12_133876:-
conv3d_7_133879:
conv3d_7_133881:-
conv3d_2_133884:
conv3d_2_133886:.
conv3d_13_133889:
conv3d_13_133891:-
conv3d_8_133894:
conv3d_8_133896:-
conv3d_3_133899:
conv3d_3_133901:-
conv3d_4_133907:
conv3d_4_133909:-
conv3d_9_133912:
conv3d_9_133914:.
conv3d_14_133917:
conv3d_14_133919:.
conv3d_25_133925:
conv3d_25_133927:.
conv3d_20_133930:
conv3d_20_133932:.
conv3d_15_133935:
conv3d_15_133937:.
conv3d_26_133943:
conv3d_26_133945:.
conv3d_21_133948:
conv3d_21_133950:.
conv3d_16_133953:
conv3d_16_133955:.
conv3d_27_133958:
conv3d_27_133960:.
conv3d_22_133963:
conv3d_22_133965:.
conv3d_17_133968:
conv3d_17_133970:.
conv3d_28_133976:
conv3d_28_133978:.
conv3d_23_133981:
conv3d_23_133983:.
conv3d_18_133986:
conv3d_18_133988:.
conv3d_19_133991:
conv3d_19_133993:.
conv3d_24_133996:
conv3d_24_133998:.
conv3d_29_134001:
conv3d_29_134003:.
conv3d_30_134007:
conv3d_30_134009:
identityЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ!conv3d_10/StatefulPartitionedCallЂ!conv3d_11/StatefulPartitionedCallЂ!conv3d_12/StatefulPartitionedCallЂ!conv3d_13/StatefulPartitionedCallЂ!conv3d_14/StatefulPartitionedCallЂ!conv3d_15/StatefulPartitionedCallЂ!conv3d_16/StatefulPartitionedCallЂ!conv3d_17/StatefulPartitionedCallЂ!conv3d_18/StatefulPartitionedCallЂ!conv3d_19/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂ!conv3d_20/StatefulPartitionedCallЂ!conv3d_21/StatefulPartitionedCallЂ!conv3d_22/StatefulPartitionedCallЂ!conv3d_23/StatefulPartitionedCallЂ!conv3d_24/StatefulPartitionedCallЂ!conv3d_25/StatefulPartitionedCallЂ!conv3d_26/StatefulPartitionedCallЂ!conv3d_27/StatefulPartitionedCallЂ!conv3d_28/StatefulPartitionedCallЂ!conv3d_29/StatefulPartitionedCallЂ conv3d_3/StatefulPartitionedCallЂ!conv3d_30/StatefulPartitionedCallЂ conv3d_4/StatefulPartitionedCallЂ conv3d_5/StatefulPartitionedCallЂ conv3d_6/StatefulPartitionedCallЂ conv3d_7/StatefulPartitionedCallЂ conv3d_8/StatefulPartitionedCallЂ conv3d_9/StatefulPartitionedCallЦ
reshape/PartitionedCallPartitionedCallu_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_131554Ъ
reshape_1/PartitionedCallPartitionedCallv_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_131571Ъ
reshape_2/PartitionedCallPartitionedCallw_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_131588Г
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_131598Ѓ
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_10_133841conv3d_10_133843*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_131611
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_5_133846conv3d_5_133848*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_131628
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_133851conv3d_133853*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_131645Љ
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_133856conv3d_11_133858*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_131662Є
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_133861conv3d_6_133863*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_131679Ђ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_133866conv3d_1_133868*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_131696ћ
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131706њ
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131712і
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131718Ї
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_133874conv3d_12_133876*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_131731Ѓ
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_133879conv3d_7_133881*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_131748Ё
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_133884conv3d_2_133886*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_131765Љ
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_133889conv3d_13_133891*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_131782Є
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_133894conv3d_8_133896*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_131799Є
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_133899conv3d_3_133901*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_131816ћ
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131826њ
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131832њ
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131838Ѓ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_133907conv3d_4_133909*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_131851Ѓ
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_133912conv3d_9_133914*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_131868Ї
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_133917conv3d_14_133919*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_131885Л
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_131899ж
reshape_6/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_131913ч
reshape_7/PartitionedCallPartitionedCall"reshape_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_7_layer_call_and_return_conditional_losses_131930Ё
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_25_133925conv3d_25_133927*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_25_layer_call_and_return_conditional_losses_131943Ё
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_20_133930conv3d_20_133932*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_20_layer_call_and_return_conditional_losses_131960Ё
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_15_133935conv3d_15_133937*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_131977ћ
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_132020ћ
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_132059ї
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_132098Ї
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_133943conv3d_26_133945*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_26_layer_call_and_return_conditional_losses_132111Ї
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_133948conv3d_21_133950*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_21_layer_call_and_return_conditional_losses_132128Ѕ
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_133953conv3d_16_133955*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_132145Љ
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_133958conv3d_27_133960*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_27_layer_call_and_return_conditional_losses_132162Љ
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_133963conv3d_22_133965*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_22_layer_call_and_return_conditional_losses_132179Љ
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_133968conv3d_17_133970*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_132196ћ
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_132263ћ
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_132326ћ
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_132389Ї
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_133976conv3d_28_133978*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_28_layer_call_and_return_conditional_losses_132402Ї
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_133981conv3d_23_133983*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_23_layer_call_and_return_conditional_losses_132419Ї
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_133986conv3d_18_133988*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_132436Љ
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_133991conv3d_19_133993*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_132453Љ
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_133996conv3d_24_133998*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_24_layer_call_and_return_conditional_losses_132470Љ
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_134001conv3d_29_134003*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_29_layer_call_and_return_conditional_losses_132487С
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_132501
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_134007conv3d_30_134009*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_30_layer_call_and_return_conditional_losses_132513
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2F
!conv3d_20/StatefulPartitionedCall!conv3d_20/StatefulPartitionedCall2F
!conv3d_21/StatefulPartitionedCall!conv3d_21/StatefulPartitionedCall2F
!conv3d_22/StatefulPartitionedCall!conv3d_22/StatefulPartitionedCall2F
!conv3d_23/StatefulPartitionedCall!conv3d_23/StatefulPartitionedCall2F
!conv3d_24/StatefulPartitionedCall!conv3d_24/StatefulPartitionedCall2F
!conv3d_25/StatefulPartitionedCall!conv3d_25/StatefulPartitionedCall2F
!conv3d_26/StatefulPartitionedCall!conv3d_26/StatefulPartitionedCall2F
!conv3d_27/StatefulPartitionedCall!conv3d_27/StatefulPartitionedCall2F
!conv3d_28/StatefulPartitionedCall!conv3d_28/StatefulPartitionedCall2F
!conv3d_29/StatefulPartitionedCall!conv3d_29/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2F
!conv3d_30/StatefulPartitionedCall!conv3d_30/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ   

_user_specified_nameu_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namev_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namew_vel

Ђ
)__inference_conv3d_4_layer_call_fn_135970

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_131851{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_5_layer_call_fn_135630

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_131628{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
У
a
E__inference_reshape_7_layer_call_and_return_conditional_losses_136073

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџd
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
џ
B__inference_conv3d_layer_call_and_return_conditional_losses_131645

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_27_layer_call_and_return_conditional_losses_136379

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
!
Ё
$__inference_signature_wrapper_134152	
u_vel	
v_vel	
w_vel%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:(

unknown_37:

unknown_38:(

unknown_39:

unknown_40:(

unknown_41:

unknown_42:(

unknown_43:

unknown_44:(

unknown_45:

unknown_46:(

unknown_47:

unknown_48:(

unknown_49:

unknown_50:(

unknown_51:

unknown_52:(

unknown_53:

unknown_54:(

unknown_55:

unknown_56:(

unknown_57:

unknown_58:(

unknown_59:

unknown_60:
identityЂStatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_131456{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ   

_user_specified_nameu_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namev_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namew_vel

Ѓ
*__inference_conv3d_20_layer_call_fn_136102

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_20_layer_call_and_return_conditional_losses_131960{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
І

D__inference_conv3d_1_layer_call_and_return_conditional_losses_131696

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_2_layer_call_fn_135790

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_131765{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_16_layer_call_and_return_conditional_losses_132145

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_12_layer_call_and_return_conditional_losses_135841

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_25_layer_call_fn_136122

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_25_layer_call_and_return_conditional_losses_131943{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131489

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_21_layer_call_fn_136288

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_21_layer_call_and_return_conditional_losses_132128{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
юо
ф
A__inference_model_layer_call_and_return_conditional_losses_133393

inputs
inputs_1
inputs_2.
conv3d_10_133221:
conv3d_10_133223:-
conv3d_5_133226:
conv3d_5_133228:+
conv3d_133231:
conv3d_133233:.
conv3d_11_133236:
conv3d_11_133238:-
conv3d_6_133241:
conv3d_6_133243:-
conv3d_1_133246:
conv3d_1_133248:.
conv3d_12_133254:
conv3d_12_133256:-
conv3d_7_133259:
conv3d_7_133261:-
conv3d_2_133264:
conv3d_2_133266:.
conv3d_13_133269:
conv3d_13_133271:-
conv3d_8_133274:
conv3d_8_133276:-
conv3d_3_133279:
conv3d_3_133281:-
conv3d_4_133287:
conv3d_4_133289:-
conv3d_9_133292:
conv3d_9_133294:.
conv3d_14_133297:
conv3d_14_133299:.
conv3d_25_133305:
conv3d_25_133307:.
conv3d_20_133310:
conv3d_20_133312:.
conv3d_15_133315:
conv3d_15_133317:.
conv3d_26_133323:
conv3d_26_133325:.
conv3d_21_133328:
conv3d_21_133330:.
conv3d_16_133333:
conv3d_16_133335:.
conv3d_27_133338:
conv3d_27_133340:.
conv3d_22_133343:
conv3d_22_133345:.
conv3d_17_133348:
conv3d_17_133350:.
conv3d_28_133356:
conv3d_28_133358:.
conv3d_23_133361:
conv3d_23_133363:.
conv3d_18_133366:
conv3d_18_133368:.
conv3d_19_133371:
conv3d_19_133373:.
conv3d_24_133376:
conv3d_24_133378:.
conv3d_29_133381:
conv3d_29_133383:.
conv3d_30_133387:
conv3d_30_133389:
identityЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ!conv3d_10/StatefulPartitionedCallЂ!conv3d_11/StatefulPartitionedCallЂ!conv3d_12/StatefulPartitionedCallЂ!conv3d_13/StatefulPartitionedCallЂ!conv3d_14/StatefulPartitionedCallЂ!conv3d_15/StatefulPartitionedCallЂ!conv3d_16/StatefulPartitionedCallЂ!conv3d_17/StatefulPartitionedCallЂ!conv3d_18/StatefulPartitionedCallЂ!conv3d_19/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂ!conv3d_20/StatefulPartitionedCallЂ!conv3d_21/StatefulPartitionedCallЂ!conv3d_22/StatefulPartitionedCallЂ!conv3d_23/StatefulPartitionedCallЂ!conv3d_24/StatefulPartitionedCallЂ!conv3d_25/StatefulPartitionedCallЂ!conv3d_26/StatefulPartitionedCallЂ!conv3d_27/StatefulPartitionedCallЂ!conv3d_28/StatefulPartitionedCallЂ!conv3d_29/StatefulPartitionedCallЂ conv3d_3/StatefulPartitionedCallЂ!conv3d_30/StatefulPartitionedCallЂ conv3d_4/StatefulPartitionedCallЂ conv3d_5/StatefulPartitionedCallЂ conv3d_6/StatefulPartitionedCallЂ conv3d_7/StatefulPartitionedCallЂ conv3d_8/StatefulPartitionedCallЂ conv3d_9/StatefulPartitionedCallЧ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_131554Э
reshape_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_131571Э
reshape_2/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_131588Г
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_131598Ѓ
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_10_133221conv3d_10_133223*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_131611
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_5_133226conv3d_5_133228*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_131628
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_133231conv3d_133233*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_131645Љ
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_133236conv3d_11_133238*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_131662Є
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_133241conv3d_6_133243*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_131679Ђ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_133246conv3d_1_133248*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_131696ћ
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131706њ
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131712і
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131718Ї
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_133254conv3d_12_133256*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_131731Ѓ
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_133259conv3d_7_133261*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_131748Ё
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_133264conv3d_2_133266*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_131765Љ
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_133269conv3d_13_133271*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_131782Є
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_133274conv3d_8_133276*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_131799Є
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_133279conv3d_3_133281*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_131816ћ
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131826њ
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131832њ
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131838Ѓ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_133287conv3d_4_133289*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_131851Ѓ
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_133292conv3d_9_133294*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_131868Ї
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_133297conv3d_14_133299*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_131885Л
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_131899ж
reshape_6/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_131913ч
reshape_7/PartitionedCallPartitionedCall"reshape_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_7_layer_call_and_return_conditional_losses_131930Ё
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_25_133305conv3d_25_133307*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_25_layer_call_and_return_conditional_losses_131943Ё
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_20_133310conv3d_20_133312*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_20_layer_call_and_return_conditional_losses_131960Ё
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_15_133315conv3d_15_133317*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_131977ћ
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_132020ћ
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_132059ї
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_132098Ї
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_133323conv3d_26_133325*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_26_layer_call_and_return_conditional_losses_132111Ї
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_133328conv3d_21_133330*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_21_layer_call_and_return_conditional_losses_132128Ѕ
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_133333conv3d_16_133335*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_132145Љ
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_133338conv3d_27_133340*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_27_layer_call_and_return_conditional_losses_132162Љ
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_133343conv3d_22_133345*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_22_layer_call_and_return_conditional_losses_132179Љ
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_133348conv3d_17_133350*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_132196ћ
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_132263ћ
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_132326ћ
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_132389Ї
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_133356conv3d_28_133358*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_28_layer_call_and_return_conditional_losses_132402Ї
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_133361conv3d_23_133363*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_23_layer_call_and_return_conditional_losses_132419Ї
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_133366conv3d_18_133368*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_132436Љ
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_133371conv3d_19_133373*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_132453Љ
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_133376conv3d_24_133378*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_24_layer_call_and_return_conditional_losses_132470Љ
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_133381conv3d_29_133383*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_29_layer_call_and_return_conditional_losses_132487С
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_132501
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_133387conv3d_30_133389*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_30_layer_call_and_return_conditional_losses_132513
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2F
!conv3d_20/StatefulPartitionedCall!conv3d_20/StatefulPartitionedCall2F
!conv3d_21/StatefulPartitionedCall!conv3d_21/StatefulPartitionedCall2F
!conv3d_22/StatefulPartitionedCall!conv3d_22/StatefulPartitionedCall2F
!conv3d_23/StatefulPartitionedCall!conv3d_23/StatefulPartitionedCall2F
!conv3d_24/StatefulPartitionedCall!conv3d_24/StatefulPartitionedCall2F
!conv3d_25/StatefulPartitionedCall!conv3d_25/StatefulPartitionedCall2F
!conv3d_26/StatefulPartitionedCall!conv3d_26/StatefulPartitionedCall2F
!conv3d_27/StatefulPartitionedCall!conv3d_27/StatefulPartitionedCall2F
!conv3d_28/StatefulPartitionedCall!conv3d_28/StatefulPartitionedCall2F
!conv3d_29/StatefulPartitionedCall!conv3d_29/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2F
!conv3d_30/StatefulPartitionedCall!conv3d_30/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs


D__inference_conv3d_4_layer_call_and_return_conditional_losses_131851

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџc
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131712

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_24_layer_call_and_return_conditional_losses_132470

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї!
Ѓ
&__inference_model_layer_call_fn_133651	
u_vel	
v_vel	
w_vel%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:(

unknown_37:

unknown_38:(

unknown_39:

unknown_40:(

unknown_41:

unknown_42:(

unknown_43:

unknown_44:(

unknown_45:

unknown_46:(

unknown_47:

unknown_48:(

unknown_49:

unknown_50:(

unknown_51:

unknown_52:(

unknown_53:

unknown_54:(

unknown_55:

unknown_56:(

unknown_57:

unknown_58:(

unknown_59:

unknown_60:
identityЂStatefulPartitionedCallЕ	
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_133393{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ   

_user_specified_nameu_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namev_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namew_vel
и
g
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_135956

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
_
C__inference_reshape_layer_call_and_return_conditional_losses_135546

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ч


E__inference_conv3d_30_layer_call_and_return_conditional_losses_132513

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
 

G__inference_concatenate_layer_call_and_return_conditional_losses_135601
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ   c
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   :] Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/2


G__inference_concatenate_layer_call_and_return_conditional_losses_131598

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ   c
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs:[W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs:[W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_12_layer_call_and_return_conditional_losses_131731

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_19_layer_call_and_return_conditional_losses_132453

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_10_layer_call_and_return_conditional_losses_135661

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
њЯ
Ш0
A__inference_model_layer_call_and_return_conditional_losses_134970
inputs_0
inputs_1
inputs_2F
(conv3d_10_conv3d_readvariableop_resource:7
)conv3d_10_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:C
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:F
(conv3d_11_conv3d_readvariableop_resource:7
)conv3d_11_biasadd_readvariableop_resource:E
'conv3d_6_conv3d_readvariableop_resource:6
(conv3d_6_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:6
(conv3d_1_biasadd_readvariableop_resource:F
(conv3d_12_conv3d_readvariableop_resource:7
)conv3d_12_biasadd_readvariableop_resource:E
'conv3d_7_conv3d_readvariableop_resource:6
(conv3d_7_biasadd_readvariableop_resource:E
'conv3d_2_conv3d_readvariableop_resource:6
(conv3d_2_biasadd_readvariableop_resource:F
(conv3d_13_conv3d_readvariableop_resource:7
)conv3d_13_biasadd_readvariableop_resource:E
'conv3d_8_conv3d_readvariableop_resource:6
(conv3d_8_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:6
(conv3d_3_biasadd_readvariableop_resource:E
'conv3d_4_conv3d_readvariableop_resource:6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_9_conv3d_readvariableop_resource:6
(conv3d_9_biasadd_readvariableop_resource:F
(conv3d_14_conv3d_readvariableop_resource:7
)conv3d_14_biasadd_readvariableop_resource:F
(conv3d_25_conv3d_readvariableop_resource:7
)conv3d_25_biasadd_readvariableop_resource:F
(conv3d_20_conv3d_readvariableop_resource:7
)conv3d_20_biasadd_readvariableop_resource:F
(conv3d_15_conv3d_readvariableop_resource:7
)conv3d_15_biasadd_readvariableop_resource:F
(conv3d_26_conv3d_readvariableop_resource:7
)conv3d_26_biasadd_readvariableop_resource:F
(conv3d_21_conv3d_readvariableop_resource:7
)conv3d_21_biasadd_readvariableop_resource:F
(conv3d_16_conv3d_readvariableop_resource:7
)conv3d_16_biasadd_readvariableop_resource:F
(conv3d_27_conv3d_readvariableop_resource:7
)conv3d_27_biasadd_readvariableop_resource:F
(conv3d_22_conv3d_readvariableop_resource:7
)conv3d_22_biasadd_readvariableop_resource:F
(conv3d_17_conv3d_readvariableop_resource:7
)conv3d_17_biasadd_readvariableop_resource:F
(conv3d_28_conv3d_readvariableop_resource:7
)conv3d_28_biasadd_readvariableop_resource:F
(conv3d_23_conv3d_readvariableop_resource:7
)conv3d_23_biasadd_readvariableop_resource:F
(conv3d_18_conv3d_readvariableop_resource:7
)conv3d_18_biasadd_readvariableop_resource:F
(conv3d_19_conv3d_readvariableop_resource:7
)conv3d_19_biasadd_readvariableop_resource:F
(conv3d_24_conv3d_readvariableop_resource:7
)conv3d_24_biasadd_readvariableop_resource:F
(conv3d_29_conv3d_readvariableop_resource:7
)conv3d_29_biasadd_readvariableop_resource:F
(conv3d_30_conv3d_readvariableop_resource:7
)conv3d_30_biasadd_readvariableop_resource:
identityЂconv3d/BiasAdd/ReadVariableOpЂconv3d/Conv3D/ReadVariableOpЂconv3d_1/BiasAdd/ReadVariableOpЂconv3d_1/Conv3D/ReadVariableOpЂ conv3d_10/BiasAdd/ReadVariableOpЂconv3d_10/Conv3D/ReadVariableOpЂ conv3d_11/BiasAdd/ReadVariableOpЂconv3d_11/Conv3D/ReadVariableOpЂ conv3d_12/BiasAdd/ReadVariableOpЂconv3d_12/Conv3D/ReadVariableOpЂ conv3d_13/BiasAdd/ReadVariableOpЂconv3d_13/Conv3D/ReadVariableOpЂ conv3d_14/BiasAdd/ReadVariableOpЂconv3d_14/Conv3D/ReadVariableOpЂ conv3d_15/BiasAdd/ReadVariableOpЂconv3d_15/Conv3D/ReadVariableOpЂ conv3d_16/BiasAdd/ReadVariableOpЂconv3d_16/Conv3D/ReadVariableOpЂ conv3d_17/BiasAdd/ReadVariableOpЂconv3d_17/Conv3D/ReadVariableOpЂ conv3d_18/BiasAdd/ReadVariableOpЂconv3d_18/Conv3D/ReadVariableOpЂ conv3d_19/BiasAdd/ReadVariableOpЂconv3d_19/Conv3D/ReadVariableOpЂconv3d_2/BiasAdd/ReadVariableOpЂconv3d_2/Conv3D/ReadVariableOpЂ conv3d_20/BiasAdd/ReadVariableOpЂconv3d_20/Conv3D/ReadVariableOpЂ conv3d_21/BiasAdd/ReadVariableOpЂconv3d_21/Conv3D/ReadVariableOpЂ conv3d_22/BiasAdd/ReadVariableOpЂconv3d_22/Conv3D/ReadVariableOpЂ conv3d_23/BiasAdd/ReadVariableOpЂconv3d_23/Conv3D/ReadVariableOpЂ conv3d_24/BiasAdd/ReadVariableOpЂconv3d_24/Conv3D/ReadVariableOpЂ conv3d_25/BiasAdd/ReadVariableOpЂconv3d_25/Conv3D/ReadVariableOpЂ conv3d_26/BiasAdd/ReadVariableOpЂconv3d_26/Conv3D/ReadVariableOpЂ conv3d_27/BiasAdd/ReadVariableOpЂconv3d_27/Conv3D/ReadVariableOpЂ conv3d_28/BiasAdd/ReadVariableOpЂconv3d_28/Conv3D/ReadVariableOpЂ conv3d_29/BiasAdd/ReadVariableOpЂconv3d_29/Conv3D/ReadVariableOpЂconv3d_3/BiasAdd/ReadVariableOpЂconv3d_3/Conv3D/ReadVariableOpЂ conv3d_30/BiasAdd/ReadVariableOpЂconv3d_30/Conv3D/ReadVariableOpЂconv3d_4/BiasAdd/ReadVariableOpЂconv3d_4/Conv3D/ReadVariableOpЂconv3d_5/BiasAdd/ReadVariableOpЂconv3d_5/Conv3D/ReadVariableOpЂconv3d_6/BiasAdd/ReadVariableOpЂconv3d_6/Conv3D/ReadVariableOpЂconv3d_7/BiasAdd/ReadVariableOpЂconv3d_7/Conv3D/ReadVariableOpЂconv3d_8/BiasAdd/ReadVariableOpЂconv3d_8/Conv3D/ReadVariableOpЂconv3d_9/BiasAdd/ReadVariableOpЂconv3d_9/Conv3D/ReadVariableOpE
reshape/ShapeShapeinputs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Y
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :ѓ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   G
reshape_1/ShapeShapeinputs_1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :џ
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapeinputs_1 reshape_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   G
reshape_2/ShapeShapeinputs_2*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_2/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :џ
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0"reshape_2/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape_2/ReshapeReshapeinputs_2 reshape_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :й
concatenate/concatConcatV2reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ч
conv3d_10/Conv3DConv3Dconcatenate/concat:output:0'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_10/ReluReluconv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_5/Conv3DConv3Dconcatenate/concat:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0С
conv3d/Conv3DConv3Dconcatenate/concat:output:0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_11/Conv3DConv3Dconv3d_10/Relu:activations:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_11/ReluReluconv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_6/Conv3DConv3Dconv3d_5/Relu:activations:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   n
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0У
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   П
max_pooling3d_4/MaxPool3D	MaxPool3Dconv3d_11/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
О
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
М
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
conv3d_12/Conv3DConv3D"max_pooling3d_4/MaxPool3D:output:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_12/ReluReluconv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ь
conv3d_7/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ъ
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_13/Conv3DConv3Dconv3d_12/Relu:activations:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_13/ReluReluconv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_8/Conv3DConv3Dconv3d_7/Relu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_8/ReluReluconv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџП
max_pooling3d_5/MaxPool3D	MaxPool3Dconv3d_13/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
О
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
О
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ь
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_4/TanhTanhconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ь
conv3d_9/Conv3DConv3D"max_pooling3d_3/MaxPool3D:output:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_9/TanhTanhconv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
conv3d_14/Conv3DConv3D"max_pooling3d_5/MaxPool3D:output:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_14/TanhTanhconv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџt
add/addAddV2conv3d_4/Tanh:y:0conv3d_9/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџq
	add/add_1AddV2add/add:z:0conv3d_14/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџL
reshape_6/ShapeShapeadd/add_1:z:0*
T0*
_output_shapes
:g
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
reshape_6/ReshapeReshapeadd/add_1:z:0 reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
reshape_7/ShapeShapereshape_6/Reshape:output:0*
T0*
_output_shapes
:g
reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_7/strided_sliceStridedSlicereshape_7/Shape:output:0&reshape_7/strided_slice/stack:output:0(reshape_7/strided_slice/stack_1:output:0(reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :џ
reshape_7/Reshape/shapePack reshape_7/strided_slice:output:0"reshape_7/Reshape/shape/1:output:0"reshape_7/Reshape/shape/2:output:0"reshape_7/Reshape/shape/3:output:0"reshape_7/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape_7/ReshapeReshapereshape_6/Reshape:output:0 reshape_7/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_25/Conv3D/ReadVariableOpReadVariableOp(conv3d_25_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ц
conv3d_25/Conv3DConv3Dreshape_7/Reshape:output:0'conv3d_25/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_25/BiasAdd/ReadVariableOpReadVariableOp)conv3d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_25/BiasAddBiasAddconv3d_25/Conv3D:output:0(conv3d_25/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_25/ReluReluconv3d_25/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_20/Conv3D/ReadVariableOpReadVariableOp(conv3d_20_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ц
conv3d_20/Conv3DConv3Dreshape_7/Reshape:output:0'conv3d_20/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_20/BiasAdd/ReadVariableOpReadVariableOp)conv3d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_20/BiasAddBiasAddconv3d_20/Conv3D:output:0(conv3d_20/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_20/ReluReluconv3d_20/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ц
conv3d_15/Conv3DConv3Dreshape_7/Reshape:output:0'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/splitSplit(up_sampling3d_4/split/split_dim:output:0conv3d_25/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/concatConcatV2up_sampling3d_4/split:output:0up_sampling3d_4/split:output:0up_sampling3d_4/split:output:1up_sampling3d_4/split:output:1up_sampling3d_4/split:output:2up_sampling3d_4/split:output:2up_sampling3d_4/split:output:3up_sampling3d_4/split:output:3up_sampling3d_4/split:output:4up_sampling3d_4/split:output:4up_sampling3d_4/split:output:5up_sampling3d_4/split:output:5up_sampling3d_4/split:output:6up_sampling3d_4/split:output:6up_sampling3d_4/split:output:7up_sampling3d_4/split:output:7$up_sampling3d_4/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/split_1Split*up_sampling3d_4/split_1/split_dim:output:0up_sampling3d_4/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_4/concat_1ConcatV2 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:7 up_sampling3d_4/split_1:output:7&up_sampling3d_4/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_4/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/split_2Split*up_sampling3d_4/split_2/split_dim:output:0!up_sampling3d_4/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_4/concat_2ConcatV2 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:7 up_sampling3d_4/split_2:output:7&up_sampling3d_4/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0conv3d_20/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7$up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7&up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7&up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ_
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0conv3d_15/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split[
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ы
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7"up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7$up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7$up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_26/Conv3D/ReadVariableOpReadVariableOp(conv3d_26_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_26/Conv3DConv3D!up_sampling3d_4/concat_2:output:0'conv3d_26/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_26/BiasAdd/ReadVariableOpReadVariableOp)conv3d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_26/BiasAddBiasAddconv3d_26/Conv3D:output:0(conv3d_26/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_26/ReluReluconv3d_26/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_21/Conv3D/ReadVariableOpReadVariableOp(conv3d_21_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_21/Conv3DConv3D!up_sampling3d_2/concat_2:output:0'conv3d_21/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_21/BiasAdd/ReadVariableOpReadVariableOp)conv3d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_21/BiasAddBiasAddconv3d_21/Conv3D:output:0(conv3d_21/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_21/ReluReluconv3d_21/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ы
conv3d_16/Conv3DConv3Dup_sampling3d/concat_2:output:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_27/Conv3D/ReadVariableOpReadVariableOp(conv3d_27_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_27/Conv3DConv3Dconv3d_26/Relu:activations:0'conv3d_27/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_27/BiasAdd/ReadVariableOpReadVariableOp)conv3d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_27/BiasAddBiasAddconv3d_27/Conv3D:output:0(conv3d_27/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_27/ReluReluconv3d_27/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_22/Conv3D/ReadVariableOpReadVariableOp(conv3d_22_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_22/Conv3DConv3Dconv3d_21/Relu:activations:0'conv3d_22/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_22/BiasAdd/ReadVariableOpReadVariableOp)conv3d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_22/BiasAddBiasAddconv3d_22/Conv3D:output:0(conv3d_22/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_22/ReluReluconv3d_22/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_17/Conv3DConv3Dconv3d_16/Relu:activations:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_17/ReluReluconv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_5/splitSplit(up_sampling3d_5/split/split_dim:output:0conv3d_27/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :	
up_sampling3d_5/concatConcatV2up_sampling3d_5/split:output:0up_sampling3d_5/split:output:0up_sampling3d_5/split:output:1up_sampling3d_5/split:output:1up_sampling3d_5/split:output:2up_sampling3d_5/split:output:2up_sampling3d_5/split:output:3up_sampling3d_5/split:output:3up_sampling3d_5/split:output:4up_sampling3d_5/split:output:4up_sampling3d_5/split:output:5up_sampling3d_5/split:output:5up_sampling3d_5/split:output:6up_sampling3d_5/split:output:6up_sampling3d_5/split:output:7up_sampling3d_5/split:output:7up_sampling3d_5/split:output:8up_sampling3d_5/split:output:8up_sampling3d_5/split:output:9up_sampling3d_5/split:output:9up_sampling3d_5/split:output:10up_sampling3d_5/split:output:10up_sampling3d_5/split:output:11up_sampling3d_5/split:output:11up_sampling3d_5/split:output:12up_sampling3d_5/split:output:12up_sampling3d_5/split:output:13up_sampling3d_5/split:output:13up_sampling3d_5/split:output:14up_sampling3d_5/split:output:14up_sampling3d_5/split:output:15up_sampling3d_5/split:output:15$up_sampling3d_5/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ c
!up_sampling3d_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_5/split_1Split*up_sampling3d_5/split_1/split_dim:output:0up_sampling3d_5/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split_
up_sampling3d_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_5/concat_1ConcatV2 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:9 up_sampling3d_5/split_1:output:9!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:15!up_sampling3d_5/split_1:output:15&up_sampling3d_5/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  c
!up_sampling3d_5/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_5/split_2Split*up_sampling3d_5/split_2/split_dim:output:0!up_sampling3d_5/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_split_
up_sampling3d_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_5/concat_2ConcatV2 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:9 up_sampling3d_5/split_2:output:9!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:15!up_sampling3d_5/split_2:output:15&up_sampling3d_5/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   a
up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_3/splitSplit(up_sampling3d_3/split/split_dim:output:0conv3d_22/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :	
up_sampling3d_3/concatConcatV2up_sampling3d_3/split:output:0up_sampling3d_3/split:output:0up_sampling3d_3/split:output:1up_sampling3d_3/split:output:1up_sampling3d_3/split:output:2up_sampling3d_3/split:output:2up_sampling3d_3/split:output:3up_sampling3d_3/split:output:3up_sampling3d_3/split:output:4up_sampling3d_3/split:output:4up_sampling3d_3/split:output:5up_sampling3d_3/split:output:5up_sampling3d_3/split:output:6up_sampling3d_3/split:output:6up_sampling3d_3/split:output:7up_sampling3d_3/split:output:7up_sampling3d_3/split:output:8up_sampling3d_3/split:output:8up_sampling3d_3/split:output:9up_sampling3d_3/split:output:9up_sampling3d_3/split:output:10up_sampling3d_3/split:output:10up_sampling3d_3/split:output:11up_sampling3d_3/split:output:11up_sampling3d_3/split:output:12up_sampling3d_3/split:output:12up_sampling3d_3/split:output:13up_sampling3d_3/split:output:13up_sampling3d_3/split:output:14up_sampling3d_3/split:output:14up_sampling3d_3/split:output:15up_sampling3d_3/split:output:15$up_sampling3d_3/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ c
!up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_3/split_1Split*up_sampling3d_3/split_1/split_dim:output:0up_sampling3d_3/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split_
up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_3/concat_1ConcatV2 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:9 up_sampling3d_3/split_1:output:9!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:15&up_sampling3d_3/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  c
!up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_3/split_2Split*up_sampling3d_3/split_2/split_dim:output:0!up_sampling3d_3/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_split_
up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_3/concat_2ConcatV2 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:9 up_sampling3d_3/split_2:output:9!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:15&up_sampling3d_3/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   a
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0conv3d_17/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :	
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15$up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ c
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split_
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15&up_sampling3d_1/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  c
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_split_
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15&up_sampling3d_1/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_28/Conv3D/ReadVariableOpReadVariableOp(conv3d_28_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_28/Conv3DConv3D!up_sampling3d_5/concat_2:output:0'conv3d_28/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_28/BiasAdd/ReadVariableOpReadVariableOp)conv3d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_28/BiasAddBiasAddconv3d_28/Conv3D:output:0(conv3d_28/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_28/ReluReluconv3d_28/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_23/Conv3D/ReadVariableOpReadVariableOp(conv3d_23_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_23/Conv3DConv3D!up_sampling3d_3/concat_2:output:0'conv3d_23/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_23/BiasAdd/ReadVariableOpReadVariableOp)conv3d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_23/BiasAddBiasAddconv3d_23/Conv3D:output:0(conv3d_23/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_23/ReluReluconv3d_23/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_18/Conv3D/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_18/Conv3DConv3D!up_sampling3d_1/concat_2:output:0'conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_18/BiasAdd/ReadVariableOpReadVariableOp)conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_18/BiasAddBiasAddconv3d_18/Conv3D:output:0(conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_18/ReluReluconv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_19/Conv3D/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_19/Conv3DConv3Dconv3d_18/Relu:activations:0'conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_19/BiasAdd/ReadVariableOpReadVariableOp)conv3d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_19/BiasAddBiasAddconv3d_19/Conv3D:output:0(conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_19/ReluReluconv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_24/Conv3D/ReadVariableOpReadVariableOp(conv3d_24_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_24/Conv3DConv3Dconv3d_23/Relu:activations:0'conv3d_24/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_24/BiasAdd/ReadVariableOpReadVariableOp)conv3d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_24/BiasAddBiasAddconv3d_24/Conv3D:output:0(conv3d_24/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_24/ReluReluconv3d_24/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_29/Conv3D/ReadVariableOpReadVariableOp(conv3d_29_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_29/Conv3DConv3Dconv3d_28/Relu:activations:0'conv3d_29/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_29/BiasAdd/ReadVariableOpReadVariableOp)conv3d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_29/BiasAddBiasAddconv3d_29/Conv3D:output:0(conv3d_29/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_29/ReluReluconv3d_29/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
	add_1/addAddV2conv3d_19/Relu:activations:0conv3d_24/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
add_1/add_1AddV2add_1/add:z:0conv3d_29/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_30/Conv3D/ReadVariableOpReadVariableOp(conv3d_30_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Л
conv3d_30/Conv3DConv3Dadd_1/add_1:z:0'conv3d_30/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_30/BiasAdd/ReadVariableOpReadVariableOp)conv3d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_30/BiasAddBiasAddconv3d_30/Conv3D:output:0(conv3d_30/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   u
IdentityIdentityconv3d_30/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp!^conv3d_18/BiasAdd/ReadVariableOp ^conv3d_18/Conv3D/ReadVariableOp!^conv3d_19/BiasAdd/ReadVariableOp ^conv3d_19/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp!^conv3d_20/BiasAdd/ReadVariableOp ^conv3d_20/Conv3D/ReadVariableOp!^conv3d_21/BiasAdd/ReadVariableOp ^conv3d_21/Conv3D/ReadVariableOp!^conv3d_22/BiasAdd/ReadVariableOp ^conv3d_22/Conv3D/ReadVariableOp!^conv3d_23/BiasAdd/ReadVariableOp ^conv3d_23/Conv3D/ReadVariableOp!^conv3d_24/BiasAdd/ReadVariableOp ^conv3d_24/Conv3D/ReadVariableOp!^conv3d_25/BiasAdd/ReadVariableOp ^conv3d_25/Conv3D/ReadVariableOp!^conv3d_26/BiasAdd/ReadVariableOp ^conv3d_26/Conv3D/ReadVariableOp!^conv3d_27/BiasAdd/ReadVariableOp ^conv3d_27/Conv3D/ReadVariableOp!^conv3d_28/BiasAdd/ReadVariableOp ^conv3d_28/Conv3D/ReadVariableOp!^conv3d_29/BiasAdd/ReadVariableOp ^conv3d_29/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp!^conv3d_30/BiasAdd/ReadVariableOp ^conv3d_30/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp2D
 conv3d_11/BiasAdd/ReadVariableOp conv3d_11/BiasAdd/ReadVariableOp2B
conv3d_11/Conv3D/ReadVariableOpconv3d_11/Conv3D/ReadVariableOp2D
 conv3d_12/BiasAdd/ReadVariableOp conv3d_12/BiasAdd/ReadVariableOp2B
conv3d_12/Conv3D/ReadVariableOpconv3d_12/Conv3D/ReadVariableOp2D
 conv3d_13/BiasAdd/ReadVariableOp conv3d_13/BiasAdd/ReadVariableOp2B
conv3d_13/Conv3D/ReadVariableOpconv3d_13/Conv3D/ReadVariableOp2D
 conv3d_14/BiasAdd/ReadVariableOp conv3d_14/BiasAdd/ReadVariableOp2B
conv3d_14/Conv3D/ReadVariableOpconv3d_14/Conv3D/ReadVariableOp2D
 conv3d_15/BiasAdd/ReadVariableOp conv3d_15/BiasAdd/ReadVariableOp2B
conv3d_15/Conv3D/ReadVariableOpconv3d_15/Conv3D/ReadVariableOp2D
 conv3d_16/BiasAdd/ReadVariableOp conv3d_16/BiasAdd/ReadVariableOp2B
conv3d_16/Conv3D/ReadVariableOpconv3d_16/Conv3D/ReadVariableOp2D
 conv3d_17/BiasAdd/ReadVariableOp conv3d_17/BiasAdd/ReadVariableOp2B
conv3d_17/Conv3D/ReadVariableOpconv3d_17/Conv3D/ReadVariableOp2D
 conv3d_18/BiasAdd/ReadVariableOp conv3d_18/BiasAdd/ReadVariableOp2B
conv3d_18/Conv3D/ReadVariableOpconv3d_18/Conv3D/ReadVariableOp2D
 conv3d_19/BiasAdd/ReadVariableOp conv3d_19/BiasAdd/ReadVariableOp2B
conv3d_19/Conv3D/ReadVariableOpconv3d_19/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2D
 conv3d_20/BiasAdd/ReadVariableOp conv3d_20/BiasAdd/ReadVariableOp2B
conv3d_20/Conv3D/ReadVariableOpconv3d_20/Conv3D/ReadVariableOp2D
 conv3d_21/BiasAdd/ReadVariableOp conv3d_21/BiasAdd/ReadVariableOp2B
conv3d_21/Conv3D/ReadVariableOpconv3d_21/Conv3D/ReadVariableOp2D
 conv3d_22/BiasAdd/ReadVariableOp conv3d_22/BiasAdd/ReadVariableOp2B
conv3d_22/Conv3D/ReadVariableOpconv3d_22/Conv3D/ReadVariableOp2D
 conv3d_23/BiasAdd/ReadVariableOp conv3d_23/BiasAdd/ReadVariableOp2B
conv3d_23/Conv3D/ReadVariableOpconv3d_23/Conv3D/ReadVariableOp2D
 conv3d_24/BiasAdd/ReadVariableOp conv3d_24/BiasAdd/ReadVariableOp2B
conv3d_24/Conv3D/ReadVariableOpconv3d_24/Conv3D/ReadVariableOp2D
 conv3d_25/BiasAdd/ReadVariableOp conv3d_25/BiasAdd/ReadVariableOp2B
conv3d_25/Conv3D/ReadVariableOpconv3d_25/Conv3D/ReadVariableOp2D
 conv3d_26/BiasAdd/ReadVariableOp conv3d_26/BiasAdd/ReadVariableOp2B
conv3d_26/Conv3D/ReadVariableOpconv3d_26/Conv3D/ReadVariableOp2D
 conv3d_27/BiasAdd/ReadVariableOp conv3d_27/BiasAdd/ReadVariableOp2B
conv3d_27/Conv3D/ReadVariableOpconv3d_27/Conv3D/ReadVariableOp2D
 conv3d_28/BiasAdd/ReadVariableOp conv3d_28/BiasAdd/ReadVariableOp2B
conv3d_28/Conv3D/ReadVariableOpconv3d_28/Conv3D/ReadVariableOp2D
 conv3d_29/BiasAdd/ReadVariableOp conv3d_29/BiasAdd/ReadVariableOp2B
conv3d_29/Conv3D/ReadVariableOpconv3d_29/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2D
 conv3d_30/BiasAdd/ReadVariableOp conv3d_30/BiasAdd/ReadVariableOp2B
conv3d_30/Conv3D/ReadVariableOpconv3d_30/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:Y U
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/2
Ї

E__inference_conv3d_11_layer_call_and_return_conditional_losses_135721

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
І

D__inference_conv3d_3_layer_call_and_return_conditional_losses_135861

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
И%
g
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_132263

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :г
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :т
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
g
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_136217

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
L
0__inference_max_pooling3d_2_layer_call_fn_135746

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131477
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_7_layer_call_fn_135810

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_131748{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_6_layer_call_fn_135690

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_131679{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_13_layer_call_and_return_conditional_losses_131782

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
р
L
0__inference_up_sampling3d_2_layer_call_fn_136180

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_132059l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_17_layer_call_and_return_conditional_losses_136339

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
L
0__inference_max_pooling3d_1_layer_call_fn_135906

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131501
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И%
g
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_132389

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :г
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :т
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_13_layer_call_and_return_conditional_losses_135901

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131513

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_9_layer_call_fn_135990

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_131868{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
F
*__inference_reshape_2_layer_call_fn_135571

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_131588l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_135921

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
О
F
*__inference_reshape_6_layer_call_fn_136041

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_131913a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
И%
g
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_136511

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :г
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :т
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_15_layer_call_and_return_conditional_losses_131977

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
w
?__inference_add_layer_call_and_return_conditional_losses_131899

inputs
inputs_1
inputs_2
identity\
addAddV2inputsinputs_1*
T0*3
_output_shapes!
:џџџџџџџџџ_
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:џџџџџџџџџ]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs:[W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs:[W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
р
L
0__inference_max_pooling3d_3_layer_call_fn_135931

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131832l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
м
J
.__inference_max_pooling3d_layer_call_fn_135731

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131718l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
р
L
0__inference_max_pooling3d_2_layer_call_fn_135751

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131712l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_21_layer_call_and_return_conditional_losses_136299

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
І

D__inference_conv3d_5_layer_call_and_return_conditional_losses_135641

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ш
D
(__inference_reshape_layer_call_fn_135531

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_131554l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
І

D__inference_conv3d_7_layer_call_and_return_conditional_losses_135821

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
І

D__inference_conv3d_3_layer_call_and_return_conditional_losses_131816

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_27_layer_call_fn_136368

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_27_layer_call_and_return_conditional_losses_132162{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131832

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
e
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_132098

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
џ
B__inference_conv3d_layer_call_and_return_conditional_losses_135621

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
І

D__inference_conv3d_2_layer_call_and_return_conditional_losses_135801

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131706

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_17_layer_call_and_return_conditional_losses_132196

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
І

D__inference_conv3d_6_layer_call_and_return_conditional_losses_135701

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs


D__inference_conv3d_9_layer_call_and_return_conditional_losses_131868

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџc
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
И%
g
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_132326

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :г
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :т
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_135736

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_1_layer_call_fn_135670

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_131696{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
р
L
0__inference_up_sampling3d_4_layer_call_fn_136222

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_132020l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_27_layer_call_and_return_conditional_losses_132162

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_12_layer_call_fn_135830

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_131731{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131477

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_16_layer_call_fn_136268

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_132145{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
g
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_132020

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_15_layer_call_fn_136082

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_131977{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_11_layer_call_fn_135710

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_131662{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї!
Ѓ
&__inference_model_layer_call_fn_132647	
u_vel	
v_vel	
w_vel%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:(

unknown_37:

unknown_38:(

unknown_39:

unknown_40:(

unknown_41:

unknown_42:(

unknown_43:

unknown_44:(

unknown_45:

unknown_46:(

unknown_47:

unknown_48:(

unknown_49:

unknown_50:(

unknown_51:

unknown_52:(

unknown_53:

unknown_54:(

unknown_55:

unknown_56:(

unknown_57:

unknown_58:(

unknown_59:

unknown_60:
identityЂStatefulPartitionedCallЕ	
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_132520{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ   

_user_specified_nameu_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namev_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namew_vel
І

D__inference_conv3d_1_layer_call_and_return_conditional_losses_135681

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_8_layer_call_fn_135870

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_131799{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_14_layer_call_fn_136010

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_131885{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131525

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ию
O
__inference__traced_save_137335
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop/
+savev2_conv3d_10_kernel_read_readvariableop-
)savev2_conv3d_10_bias_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop/
+savev2_conv3d_11_kernel_read_readvariableop-
)savev2_conv3d_11_bias_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop.
*savev2_conv3d_7_kernel_read_readvariableop,
(savev2_conv3d_7_bias_read_readvariableop/
+savev2_conv3d_12_kernel_read_readvariableop-
)savev2_conv3d_12_bias_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_8_kernel_read_readvariableop,
(savev2_conv3d_8_bias_read_readvariableop/
+savev2_conv3d_13_kernel_read_readvariableop-
)savev2_conv3d_13_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_conv3d_9_kernel_read_readvariableop,
(savev2_conv3d_9_bias_read_readvariableop/
+savev2_conv3d_14_kernel_read_readvariableop-
)savev2_conv3d_14_bias_read_readvariableop/
+savev2_conv3d_15_kernel_read_readvariableop-
)savev2_conv3d_15_bias_read_readvariableop/
+savev2_conv3d_20_kernel_read_readvariableop-
)savev2_conv3d_20_bias_read_readvariableop/
+savev2_conv3d_25_kernel_read_readvariableop-
)savev2_conv3d_25_bias_read_readvariableop/
+savev2_conv3d_16_kernel_read_readvariableop-
)savev2_conv3d_16_bias_read_readvariableop/
+savev2_conv3d_21_kernel_read_readvariableop-
)savev2_conv3d_21_bias_read_readvariableop/
+savev2_conv3d_26_kernel_read_readvariableop-
)savev2_conv3d_26_bias_read_readvariableop/
+savev2_conv3d_17_kernel_read_readvariableop-
)savev2_conv3d_17_bias_read_readvariableop/
+savev2_conv3d_22_kernel_read_readvariableop-
)savev2_conv3d_22_bias_read_readvariableop/
+savev2_conv3d_27_kernel_read_readvariableop-
)savev2_conv3d_27_bias_read_readvariableop/
+savev2_conv3d_18_kernel_read_readvariableop-
)savev2_conv3d_18_bias_read_readvariableop/
+savev2_conv3d_23_kernel_read_readvariableop-
)savev2_conv3d_23_bias_read_readvariableop/
+savev2_conv3d_28_kernel_read_readvariableop-
)savev2_conv3d_28_bias_read_readvariableop/
+savev2_conv3d_19_kernel_read_readvariableop-
)savev2_conv3d_19_bias_read_readvariableop/
+savev2_conv3d_24_kernel_read_readvariableop-
)savev2_conv3d_24_bias_read_readvariableop/
+savev2_conv3d_29_kernel_read_readvariableop-
)savev2_conv3d_29_bias_read_readvariableop/
+savev2_conv3d_30_kernel_read_readvariableop-
)savev2_conv3d_30_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv3d_kernel_m_read_readvariableop1
-savev2_adam_conv3d_bias_m_read_readvariableop5
1savev2_adam_conv3d_5_kernel_m_read_readvariableop3
/savev2_adam_conv3d_5_bias_m_read_readvariableop6
2savev2_adam_conv3d_10_kernel_m_read_readvariableop4
0savev2_adam_conv3d_10_bias_m_read_readvariableop5
1savev2_adam_conv3d_1_kernel_m_read_readvariableop3
/savev2_adam_conv3d_1_bias_m_read_readvariableop5
1savev2_adam_conv3d_6_kernel_m_read_readvariableop3
/savev2_adam_conv3d_6_bias_m_read_readvariableop6
2savev2_adam_conv3d_11_kernel_m_read_readvariableop4
0savev2_adam_conv3d_11_bias_m_read_readvariableop5
1savev2_adam_conv3d_2_kernel_m_read_readvariableop3
/savev2_adam_conv3d_2_bias_m_read_readvariableop5
1savev2_adam_conv3d_7_kernel_m_read_readvariableop3
/savev2_adam_conv3d_7_bias_m_read_readvariableop6
2savev2_adam_conv3d_12_kernel_m_read_readvariableop4
0savev2_adam_conv3d_12_bias_m_read_readvariableop5
1savev2_adam_conv3d_3_kernel_m_read_readvariableop3
/savev2_adam_conv3d_3_bias_m_read_readvariableop5
1savev2_adam_conv3d_8_kernel_m_read_readvariableop3
/savev2_adam_conv3d_8_bias_m_read_readvariableop6
2savev2_adam_conv3d_13_kernel_m_read_readvariableop4
0savev2_adam_conv3d_13_bias_m_read_readvariableop5
1savev2_adam_conv3d_4_kernel_m_read_readvariableop3
/savev2_adam_conv3d_4_bias_m_read_readvariableop5
1savev2_adam_conv3d_9_kernel_m_read_readvariableop3
/savev2_adam_conv3d_9_bias_m_read_readvariableop6
2savev2_adam_conv3d_14_kernel_m_read_readvariableop4
0savev2_adam_conv3d_14_bias_m_read_readvariableop6
2savev2_adam_conv3d_15_kernel_m_read_readvariableop4
0savev2_adam_conv3d_15_bias_m_read_readvariableop6
2savev2_adam_conv3d_20_kernel_m_read_readvariableop4
0savev2_adam_conv3d_20_bias_m_read_readvariableop6
2savev2_adam_conv3d_25_kernel_m_read_readvariableop4
0savev2_adam_conv3d_25_bias_m_read_readvariableop6
2savev2_adam_conv3d_16_kernel_m_read_readvariableop4
0savev2_adam_conv3d_16_bias_m_read_readvariableop6
2savev2_adam_conv3d_21_kernel_m_read_readvariableop4
0savev2_adam_conv3d_21_bias_m_read_readvariableop6
2savev2_adam_conv3d_26_kernel_m_read_readvariableop4
0savev2_adam_conv3d_26_bias_m_read_readvariableop6
2savev2_adam_conv3d_17_kernel_m_read_readvariableop4
0savev2_adam_conv3d_17_bias_m_read_readvariableop6
2savev2_adam_conv3d_22_kernel_m_read_readvariableop4
0savev2_adam_conv3d_22_bias_m_read_readvariableop6
2savev2_adam_conv3d_27_kernel_m_read_readvariableop4
0savev2_adam_conv3d_27_bias_m_read_readvariableop6
2savev2_adam_conv3d_18_kernel_m_read_readvariableop4
0savev2_adam_conv3d_18_bias_m_read_readvariableop6
2savev2_adam_conv3d_23_kernel_m_read_readvariableop4
0savev2_adam_conv3d_23_bias_m_read_readvariableop6
2savev2_adam_conv3d_28_kernel_m_read_readvariableop4
0savev2_adam_conv3d_28_bias_m_read_readvariableop6
2savev2_adam_conv3d_19_kernel_m_read_readvariableop4
0savev2_adam_conv3d_19_bias_m_read_readvariableop6
2savev2_adam_conv3d_24_kernel_m_read_readvariableop4
0savev2_adam_conv3d_24_bias_m_read_readvariableop6
2savev2_adam_conv3d_29_kernel_m_read_readvariableop4
0savev2_adam_conv3d_29_bias_m_read_readvariableop6
2savev2_adam_conv3d_30_kernel_m_read_readvariableop4
0savev2_adam_conv3d_30_bias_m_read_readvariableop3
/savev2_adam_conv3d_kernel_v_read_readvariableop1
-savev2_adam_conv3d_bias_v_read_readvariableop5
1savev2_adam_conv3d_5_kernel_v_read_readvariableop3
/savev2_adam_conv3d_5_bias_v_read_readvariableop6
2savev2_adam_conv3d_10_kernel_v_read_readvariableop4
0savev2_adam_conv3d_10_bias_v_read_readvariableop5
1savev2_adam_conv3d_1_kernel_v_read_readvariableop3
/savev2_adam_conv3d_1_bias_v_read_readvariableop5
1savev2_adam_conv3d_6_kernel_v_read_readvariableop3
/savev2_adam_conv3d_6_bias_v_read_readvariableop6
2savev2_adam_conv3d_11_kernel_v_read_readvariableop4
0savev2_adam_conv3d_11_bias_v_read_readvariableop5
1savev2_adam_conv3d_2_kernel_v_read_readvariableop3
/savev2_adam_conv3d_2_bias_v_read_readvariableop5
1savev2_adam_conv3d_7_kernel_v_read_readvariableop3
/savev2_adam_conv3d_7_bias_v_read_readvariableop6
2savev2_adam_conv3d_12_kernel_v_read_readvariableop4
0savev2_adam_conv3d_12_bias_v_read_readvariableop5
1savev2_adam_conv3d_3_kernel_v_read_readvariableop3
/savev2_adam_conv3d_3_bias_v_read_readvariableop5
1savev2_adam_conv3d_8_kernel_v_read_readvariableop3
/savev2_adam_conv3d_8_bias_v_read_readvariableop6
2savev2_adam_conv3d_13_kernel_v_read_readvariableop4
0savev2_adam_conv3d_13_bias_v_read_readvariableop5
1savev2_adam_conv3d_4_kernel_v_read_readvariableop3
/savev2_adam_conv3d_4_bias_v_read_readvariableop5
1savev2_adam_conv3d_9_kernel_v_read_readvariableop3
/savev2_adam_conv3d_9_bias_v_read_readvariableop6
2savev2_adam_conv3d_14_kernel_v_read_readvariableop4
0savev2_adam_conv3d_14_bias_v_read_readvariableop6
2savev2_adam_conv3d_15_kernel_v_read_readvariableop4
0savev2_adam_conv3d_15_bias_v_read_readvariableop6
2savev2_adam_conv3d_20_kernel_v_read_readvariableop4
0savev2_adam_conv3d_20_bias_v_read_readvariableop6
2savev2_adam_conv3d_25_kernel_v_read_readvariableop4
0savev2_adam_conv3d_25_bias_v_read_readvariableop6
2savev2_adam_conv3d_16_kernel_v_read_readvariableop4
0savev2_adam_conv3d_16_bias_v_read_readvariableop6
2savev2_adam_conv3d_21_kernel_v_read_readvariableop4
0savev2_adam_conv3d_21_bias_v_read_readvariableop6
2savev2_adam_conv3d_26_kernel_v_read_readvariableop4
0savev2_adam_conv3d_26_bias_v_read_readvariableop6
2savev2_adam_conv3d_17_kernel_v_read_readvariableop4
0savev2_adam_conv3d_17_bias_v_read_readvariableop6
2savev2_adam_conv3d_22_kernel_v_read_readvariableop4
0savev2_adam_conv3d_22_bias_v_read_readvariableop6
2savev2_adam_conv3d_27_kernel_v_read_readvariableop4
0savev2_adam_conv3d_27_bias_v_read_readvariableop6
2savev2_adam_conv3d_18_kernel_v_read_readvariableop4
0savev2_adam_conv3d_18_bias_v_read_readvariableop6
2savev2_adam_conv3d_23_kernel_v_read_readvariableop4
0savev2_adam_conv3d_23_bias_v_read_readvariableop6
2savev2_adam_conv3d_28_kernel_v_read_readvariableop4
0savev2_adam_conv3d_28_bias_v_read_readvariableop6
2savev2_adam_conv3d_19_kernel_v_read_readvariableop4
0savev2_adam_conv3d_19_bias_v_read_readvariableop6
2savev2_adam_conv3d_24_kernel_v_read_readvariableop4
0savev2_adam_conv3d_24_bias_v_read_readvariableop6
2savev2_adam_conv3d_29_kernel_v_read_readvariableop4
0savev2_adam_conv3d_29_bias_v_read_readvariableop6
2savev2_adam_conv3d_30_kernel_v_read_readvariableop4
0savev2_adam_conv3d_30_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: сo
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:Т*
dtype0*o
valueџnBќnТB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHі
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:Т*
dtype0*
valueBТB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B K
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop+savev2_conv3d_10_kernel_read_readvariableop)savev2_conv3d_10_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop+savev2_conv3d_11_kernel_read_readvariableop)savev2_conv3d_11_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop+savev2_conv3d_12_kernel_read_readvariableop)savev2_conv3d_12_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_8_kernel_read_readvariableop(savev2_conv3d_8_bias_read_readvariableop+savev2_conv3d_13_kernel_read_readvariableop)savev2_conv3d_13_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_9_kernel_read_readvariableop(savev2_conv3d_9_bias_read_readvariableop+savev2_conv3d_14_kernel_read_readvariableop)savev2_conv3d_14_bias_read_readvariableop+savev2_conv3d_15_kernel_read_readvariableop)savev2_conv3d_15_bias_read_readvariableop+savev2_conv3d_20_kernel_read_readvariableop)savev2_conv3d_20_bias_read_readvariableop+savev2_conv3d_25_kernel_read_readvariableop)savev2_conv3d_25_bias_read_readvariableop+savev2_conv3d_16_kernel_read_readvariableop)savev2_conv3d_16_bias_read_readvariableop+savev2_conv3d_21_kernel_read_readvariableop)savev2_conv3d_21_bias_read_readvariableop+savev2_conv3d_26_kernel_read_readvariableop)savev2_conv3d_26_bias_read_readvariableop+savev2_conv3d_17_kernel_read_readvariableop)savev2_conv3d_17_bias_read_readvariableop+savev2_conv3d_22_kernel_read_readvariableop)savev2_conv3d_22_bias_read_readvariableop+savev2_conv3d_27_kernel_read_readvariableop)savev2_conv3d_27_bias_read_readvariableop+savev2_conv3d_18_kernel_read_readvariableop)savev2_conv3d_18_bias_read_readvariableop+savev2_conv3d_23_kernel_read_readvariableop)savev2_conv3d_23_bias_read_readvariableop+savev2_conv3d_28_kernel_read_readvariableop)savev2_conv3d_28_bias_read_readvariableop+savev2_conv3d_19_kernel_read_readvariableop)savev2_conv3d_19_bias_read_readvariableop+savev2_conv3d_24_kernel_read_readvariableop)savev2_conv3d_24_bias_read_readvariableop+savev2_conv3d_29_kernel_read_readvariableop)savev2_conv3d_29_bias_read_readvariableop+savev2_conv3d_30_kernel_read_readvariableop)savev2_conv3d_30_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop2savev2_adam_conv3d_10_kernel_m_read_readvariableop0savev2_adam_conv3d_10_bias_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop1savev2_adam_conv3d_6_kernel_m_read_readvariableop/savev2_adam_conv3d_6_bias_m_read_readvariableop2savev2_adam_conv3d_11_kernel_m_read_readvariableop0savev2_adam_conv3d_11_bias_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop1savev2_adam_conv3d_7_kernel_m_read_readvariableop/savev2_adam_conv3d_7_bias_m_read_readvariableop2savev2_adam_conv3d_12_kernel_m_read_readvariableop0savev2_adam_conv3d_12_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_8_kernel_m_read_readvariableop/savev2_adam_conv3d_8_bias_m_read_readvariableop2savev2_adam_conv3d_13_kernel_m_read_readvariableop0savev2_adam_conv3d_13_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_conv3d_9_kernel_m_read_readvariableop/savev2_adam_conv3d_9_bias_m_read_readvariableop2savev2_adam_conv3d_14_kernel_m_read_readvariableop0savev2_adam_conv3d_14_bias_m_read_readvariableop2savev2_adam_conv3d_15_kernel_m_read_readvariableop0savev2_adam_conv3d_15_bias_m_read_readvariableop2savev2_adam_conv3d_20_kernel_m_read_readvariableop0savev2_adam_conv3d_20_bias_m_read_readvariableop2savev2_adam_conv3d_25_kernel_m_read_readvariableop0savev2_adam_conv3d_25_bias_m_read_readvariableop2savev2_adam_conv3d_16_kernel_m_read_readvariableop0savev2_adam_conv3d_16_bias_m_read_readvariableop2savev2_adam_conv3d_21_kernel_m_read_readvariableop0savev2_adam_conv3d_21_bias_m_read_readvariableop2savev2_adam_conv3d_26_kernel_m_read_readvariableop0savev2_adam_conv3d_26_bias_m_read_readvariableop2savev2_adam_conv3d_17_kernel_m_read_readvariableop0savev2_adam_conv3d_17_bias_m_read_readvariableop2savev2_adam_conv3d_22_kernel_m_read_readvariableop0savev2_adam_conv3d_22_bias_m_read_readvariableop2savev2_adam_conv3d_27_kernel_m_read_readvariableop0savev2_adam_conv3d_27_bias_m_read_readvariableop2savev2_adam_conv3d_18_kernel_m_read_readvariableop0savev2_adam_conv3d_18_bias_m_read_readvariableop2savev2_adam_conv3d_23_kernel_m_read_readvariableop0savev2_adam_conv3d_23_bias_m_read_readvariableop2savev2_adam_conv3d_28_kernel_m_read_readvariableop0savev2_adam_conv3d_28_bias_m_read_readvariableop2savev2_adam_conv3d_19_kernel_m_read_readvariableop0savev2_adam_conv3d_19_bias_m_read_readvariableop2savev2_adam_conv3d_24_kernel_m_read_readvariableop0savev2_adam_conv3d_24_bias_m_read_readvariableop2savev2_adam_conv3d_29_kernel_m_read_readvariableop0savev2_adam_conv3d_29_bias_m_read_readvariableop2savev2_adam_conv3d_30_kernel_m_read_readvariableop0savev2_adam_conv3d_30_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop2savev2_adam_conv3d_10_kernel_v_read_readvariableop0savev2_adam_conv3d_10_bias_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop1savev2_adam_conv3d_6_kernel_v_read_readvariableop/savev2_adam_conv3d_6_bias_v_read_readvariableop2savev2_adam_conv3d_11_kernel_v_read_readvariableop0savev2_adam_conv3d_11_bias_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop1savev2_adam_conv3d_7_kernel_v_read_readvariableop/savev2_adam_conv3d_7_bias_v_read_readvariableop2savev2_adam_conv3d_12_kernel_v_read_readvariableop0savev2_adam_conv3d_12_bias_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_8_kernel_v_read_readvariableop/savev2_adam_conv3d_8_bias_v_read_readvariableop2savev2_adam_conv3d_13_kernel_v_read_readvariableop0savev2_adam_conv3d_13_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_conv3d_9_kernel_v_read_readvariableop/savev2_adam_conv3d_9_bias_v_read_readvariableop2savev2_adam_conv3d_14_kernel_v_read_readvariableop0savev2_adam_conv3d_14_bias_v_read_readvariableop2savev2_adam_conv3d_15_kernel_v_read_readvariableop0savev2_adam_conv3d_15_bias_v_read_readvariableop2savev2_adam_conv3d_20_kernel_v_read_readvariableop0savev2_adam_conv3d_20_bias_v_read_readvariableop2savev2_adam_conv3d_25_kernel_v_read_readvariableop0savev2_adam_conv3d_25_bias_v_read_readvariableop2savev2_adam_conv3d_16_kernel_v_read_readvariableop0savev2_adam_conv3d_16_bias_v_read_readvariableop2savev2_adam_conv3d_21_kernel_v_read_readvariableop0savev2_adam_conv3d_21_bias_v_read_readvariableop2savev2_adam_conv3d_26_kernel_v_read_readvariableop0savev2_adam_conv3d_26_bias_v_read_readvariableop2savev2_adam_conv3d_17_kernel_v_read_readvariableop0savev2_adam_conv3d_17_bias_v_read_readvariableop2savev2_adam_conv3d_22_kernel_v_read_readvariableop0savev2_adam_conv3d_22_bias_v_read_readvariableop2savev2_adam_conv3d_27_kernel_v_read_readvariableop0savev2_adam_conv3d_27_bias_v_read_readvariableop2savev2_adam_conv3d_18_kernel_v_read_readvariableop0savev2_adam_conv3d_18_bias_v_read_readvariableop2savev2_adam_conv3d_23_kernel_v_read_readvariableop0savev2_adam_conv3d_23_bias_v_read_readvariableop2savev2_adam_conv3d_28_kernel_v_read_readvariableop0savev2_adam_conv3d_28_bias_v_read_readvariableop2savev2_adam_conv3d_19_kernel_v_read_readvariableop0savev2_adam_conv3d_19_bias_v_read_readvariableop2savev2_adam_conv3d_24_kernel_v_read_readvariableop0savev2_adam_conv3d_24_bias_v_read_readvariableop2savev2_adam_conv3d_29_kernel_v_read_readvariableop0savev2_adam_conv3d_29_bias_v_read_readvariableop2savev2_adam_conv3d_30_kernel_v_read_readvariableop0savev2_adam_conv3d_30_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *г
dtypesШ
Х2Т	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*г
_input_shapesС
О: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: : : : : : : ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0	,
*
_output_shapes
:: 


_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
::  

_output_shapes
::0!,
*
_output_shapes
:: "

_output_shapes
::0#,
*
_output_shapes
:: $

_output_shapes
::0%,
*
_output_shapes
:: &

_output_shapes
::0',
*
_output_shapes
:: (

_output_shapes
::0),
*
_output_shapes
:: *

_output_shapes
::0+,
*
_output_shapes
:: ,

_output_shapes
::0-,
*
_output_shapes
:: .

_output_shapes
::0/,
*
_output_shapes
:: 0

_output_shapes
::01,
*
_output_shapes
:: 2

_output_shapes
::03,
*
_output_shapes
:: 4

_output_shapes
::05,
*
_output_shapes
:: 6

_output_shapes
::07,
*
_output_shapes
:: 8

_output_shapes
::09,
*
_output_shapes
:: :

_output_shapes
::0;,
*
_output_shapes
:: <

_output_shapes
::0=,
*
_output_shapes
:: >

_output_shapes
::?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :0F,
*
_output_shapes
:: G

_output_shapes
::0H,
*
_output_shapes
:: I

_output_shapes
::0J,
*
_output_shapes
:: K

_output_shapes
::0L,
*
_output_shapes
:: M

_output_shapes
::0N,
*
_output_shapes
:: O

_output_shapes
::0P,
*
_output_shapes
:: Q

_output_shapes
::0R,
*
_output_shapes
:: S

_output_shapes
::0T,
*
_output_shapes
:: U

_output_shapes
::0V,
*
_output_shapes
:: W

_output_shapes
::0X,
*
_output_shapes
:: Y

_output_shapes
::0Z,
*
_output_shapes
:: [

_output_shapes
::0\,
*
_output_shapes
:: ]

_output_shapes
::0^,
*
_output_shapes
:: _

_output_shapes
::0`,
*
_output_shapes
:: a

_output_shapes
::0b,
*
_output_shapes
:: c

_output_shapes
::0d,
*
_output_shapes
:: e

_output_shapes
::0f,
*
_output_shapes
:: g

_output_shapes
::0h,
*
_output_shapes
:: i

_output_shapes
::0j,
*
_output_shapes
:: k

_output_shapes
::0l,
*
_output_shapes
:: m

_output_shapes
::0n,
*
_output_shapes
:: o

_output_shapes
::0p,
*
_output_shapes
:: q

_output_shapes
::0r,
*
_output_shapes
:: s

_output_shapes
::0t,
*
_output_shapes
:: u

_output_shapes
::0v,
*
_output_shapes
:: w

_output_shapes
::0x,
*
_output_shapes
:: y

_output_shapes
::0z,
*
_output_shapes
:: {

_output_shapes
::0|,
*
_output_shapes
:: }

_output_shapes
::0~,
*
_output_shapes
:: 

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1,
*
_output_shapes
::!

_output_shapes
::1 ,
*
_output_shapes
::!Ё

_output_shapes
::1Ђ,
*
_output_shapes
::!Ѓ

_output_shapes
::1Є,
*
_output_shapes
::!Ѕ

_output_shapes
::1І,
*
_output_shapes
::!Ї

_output_shapes
::1Ј,
*
_output_shapes
::!Љ

_output_shapes
::1Њ,
*
_output_shapes
::!Ћ

_output_shapes
::1Ќ,
*
_output_shapes
::!­

_output_shapes
::1Ў,
*
_output_shapes
::!Џ

_output_shapes
::1А,
*
_output_shapes
::!Б

_output_shapes
::1В,
*
_output_shapes
::!Г

_output_shapes
::1Д,
*
_output_shapes
::!Е

_output_shapes
::1Ж,
*
_output_shapes
::!З

_output_shapes
::1И,
*
_output_shapes
::!Й

_output_shapes
::1К,
*
_output_shapes
::!Л

_output_shapes
::1М,
*
_output_shapes
::!Н

_output_shapes
::1О,
*
_output_shapes
::!П

_output_shapes
::1Р,
*
_output_shapes
::!С

_output_shapes
::Т

_output_shapes
: 
Ї

E__inference_conv3d_20_layer_call_and_return_conditional_losses_131960

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
И%
g
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_136577

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :г
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :т
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
њЯ
Ш0
A__inference_model_layer_call_and_return_conditional_losses_135526
inputs_0
inputs_1
inputs_2F
(conv3d_10_conv3d_readvariableop_resource:7
)conv3d_10_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:C
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:F
(conv3d_11_conv3d_readvariableop_resource:7
)conv3d_11_biasadd_readvariableop_resource:E
'conv3d_6_conv3d_readvariableop_resource:6
(conv3d_6_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:6
(conv3d_1_biasadd_readvariableop_resource:F
(conv3d_12_conv3d_readvariableop_resource:7
)conv3d_12_biasadd_readvariableop_resource:E
'conv3d_7_conv3d_readvariableop_resource:6
(conv3d_7_biasadd_readvariableop_resource:E
'conv3d_2_conv3d_readvariableop_resource:6
(conv3d_2_biasadd_readvariableop_resource:F
(conv3d_13_conv3d_readvariableop_resource:7
)conv3d_13_biasadd_readvariableop_resource:E
'conv3d_8_conv3d_readvariableop_resource:6
(conv3d_8_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:6
(conv3d_3_biasadd_readvariableop_resource:E
'conv3d_4_conv3d_readvariableop_resource:6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_9_conv3d_readvariableop_resource:6
(conv3d_9_biasadd_readvariableop_resource:F
(conv3d_14_conv3d_readvariableop_resource:7
)conv3d_14_biasadd_readvariableop_resource:F
(conv3d_25_conv3d_readvariableop_resource:7
)conv3d_25_biasadd_readvariableop_resource:F
(conv3d_20_conv3d_readvariableop_resource:7
)conv3d_20_biasadd_readvariableop_resource:F
(conv3d_15_conv3d_readvariableop_resource:7
)conv3d_15_biasadd_readvariableop_resource:F
(conv3d_26_conv3d_readvariableop_resource:7
)conv3d_26_biasadd_readvariableop_resource:F
(conv3d_21_conv3d_readvariableop_resource:7
)conv3d_21_biasadd_readvariableop_resource:F
(conv3d_16_conv3d_readvariableop_resource:7
)conv3d_16_biasadd_readvariableop_resource:F
(conv3d_27_conv3d_readvariableop_resource:7
)conv3d_27_biasadd_readvariableop_resource:F
(conv3d_22_conv3d_readvariableop_resource:7
)conv3d_22_biasadd_readvariableop_resource:F
(conv3d_17_conv3d_readvariableop_resource:7
)conv3d_17_biasadd_readvariableop_resource:F
(conv3d_28_conv3d_readvariableop_resource:7
)conv3d_28_biasadd_readvariableop_resource:F
(conv3d_23_conv3d_readvariableop_resource:7
)conv3d_23_biasadd_readvariableop_resource:F
(conv3d_18_conv3d_readvariableop_resource:7
)conv3d_18_biasadd_readvariableop_resource:F
(conv3d_19_conv3d_readvariableop_resource:7
)conv3d_19_biasadd_readvariableop_resource:F
(conv3d_24_conv3d_readvariableop_resource:7
)conv3d_24_biasadd_readvariableop_resource:F
(conv3d_29_conv3d_readvariableop_resource:7
)conv3d_29_biasadd_readvariableop_resource:F
(conv3d_30_conv3d_readvariableop_resource:7
)conv3d_30_biasadd_readvariableop_resource:
identityЂconv3d/BiasAdd/ReadVariableOpЂconv3d/Conv3D/ReadVariableOpЂconv3d_1/BiasAdd/ReadVariableOpЂconv3d_1/Conv3D/ReadVariableOpЂ conv3d_10/BiasAdd/ReadVariableOpЂconv3d_10/Conv3D/ReadVariableOpЂ conv3d_11/BiasAdd/ReadVariableOpЂconv3d_11/Conv3D/ReadVariableOpЂ conv3d_12/BiasAdd/ReadVariableOpЂconv3d_12/Conv3D/ReadVariableOpЂ conv3d_13/BiasAdd/ReadVariableOpЂconv3d_13/Conv3D/ReadVariableOpЂ conv3d_14/BiasAdd/ReadVariableOpЂconv3d_14/Conv3D/ReadVariableOpЂ conv3d_15/BiasAdd/ReadVariableOpЂconv3d_15/Conv3D/ReadVariableOpЂ conv3d_16/BiasAdd/ReadVariableOpЂconv3d_16/Conv3D/ReadVariableOpЂ conv3d_17/BiasAdd/ReadVariableOpЂconv3d_17/Conv3D/ReadVariableOpЂ conv3d_18/BiasAdd/ReadVariableOpЂconv3d_18/Conv3D/ReadVariableOpЂ conv3d_19/BiasAdd/ReadVariableOpЂconv3d_19/Conv3D/ReadVariableOpЂconv3d_2/BiasAdd/ReadVariableOpЂconv3d_2/Conv3D/ReadVariableOpЂ conv3d_20/BiasAdd/ReadVariableOpЂconv3d_20/Conv3D/ReadVariableOpЂ conv3d_21/BiasAdd/ReadVariableOpЂconv3d_21/Conv3D/ReadVariableOpЂ conv3d_22/BiasAdd/ReadVariableOpЂconv3d_22/Conv3D/ReadVariableOpЂ conv3d_23/BiasAdd/ReadVariableOpЂconv3d_23/Conv3D/ReadVariableOpЂ conv3d_24/BiasAdd/ReadVariableOpЂconv3d_24/Conv3D/ReadVariableOpЂ conv3d_25/BiasAdd/ReadVariableOpЂconv3d_25/Conv3D/ReadVariableOpЂ conv3d_26/BiasAdd/ReadVariableOpЂconv3d_26/Conv3D/ReadVariableOpЂ conv3d_27/BiasAdd/ReadVariableOpЂconv3d_27/Conv3D/ReadVariableOpЂ conv3d_28/BiasAdd/ReadVariableOpЂconv3d_28/Conv3D/ReadVariableOpЂ conv3d_29/BiasAdd/ReadVariableOpЂconv3d_29/Conv3D/ReadVariableOpЂconv3d_3/BiasAdd/ReadVariableOpЂconv3d_3/Conv3D/ReadVariableOpЂ conv3d_30/BiasAdd/ReadVariableOpЂconv3d_30/Conv3D/ReadVariableOpЂconv3d_4/BiasAdd/ReadVariableOpЂconv3d_4/Conv3D/ReadVariableOpЂconv3d_5/BiasAdd/ReadVariableOpЂconv3d_5/Conv3D/ReadVariableOpЂconv3d_6/BiasAdd/ReadVariableOpЂconv3d_6/Conv3D/ReadVariableOpЂconv3d_7/BiasAdd/ReadVariableOpЂconv3d_7/Conv3D/ReadVariableOpЂconv3d_8/BiasAdd/ReadVariableOpЂconv3d_8/Conv3D/ReadVariableOpЂconv3d_9/BiasAdd/ReadVariableOpЂconv3d_9/Conv3D/ReadVariableOpE
reshape/ShapeShapeinputs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Y
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :ѓ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   G
reshape_1/ShapeShapeinputs_1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :џ
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapeinputs_1 reshape_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   G
reshape_2/ShapeShapeinputs_2*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : [
reshape_2/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :џ
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0"reshape_2/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape_2/ReshapeReshapeinputs_2 reshape_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :й
concatenate/concatConcatV2reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ч
conv3d_10/Conv3DConv3Dconcatenate/concat:output:0'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_10/ReluReluconv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_5/Conv3DConv3Dconcatenate/concat:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0С
conv3d/Conv3DConv3Dconcatenate/concat:output:0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_11/Conv3DConv3Dconv3d_10/Relu:activations:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_11/ReluReluconv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_6/Conv3DConv3Dconv3d_5/Relu:activations:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   n
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0У
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   П
max_pooling3d_4/MaxPool3D	MaxPool3Dconv3d_11/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
О
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
М
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
conv3d_12/Conv3DConv3D"max_pooling3d_4/MaxPool3D:output:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_12/ReluReluconv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ь
conv3d_7/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ъ
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_13/Conv3DConv3Dconv3d_12/Relu:activations:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_13/ReluReluconv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_8/Conv3DConv3Dconv3d_7/Relu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_8/ReluReluconv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Х
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџП
max_pooling3d_5/MaxPool3D	MaxPool3Dconv3d_13/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
О
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
О
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ь
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_4/TanhTanhconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ь
conv3d_9/Conv3DConv3D"max_pooling3d_3/MaxPool3D:output:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџn
conv3d_9/TanhTanhconv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
conv3d_14/Conv3DConv3D"max_pooling3d_5/MaxPool3D:output:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_14/TanhTanhconv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџt
add/addAddV2conv3d_4/Tanh:y:0conv3d_9/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџq
	add/add_1AddV2add/add:z:0conv3d_14/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџL
reshape_6/ShapeShapeadd/add_1:z:0*
T0*
_output_shapes
:g
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
reshape_6/ReshapeReshapeadd/add_1:z:0 reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
reshape_7/ShapeShapereshape_6/Reshape:output:0*
T0*
_output_shapes
:g
reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_7/strided_sliceStridedSlicereshape_7/Shape:output:0&reshape_7/strided_slice/stack:output:0(reshape_7/strided_slice/stack_1:output:0(reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :џ
reshape_7/Reshape/shapePack reshape_7/strided_slice:output:0"reshape_7/Reshape/shape/1:output:0"reshape_7/Reshape/shape/2:output:0"reshape_7/Reshape/shape/3:output:0"reshape_7/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:
reshape_7/ReshapeReshapereshape_6/Reshape:output:0 reshape_7/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_25/Conv3D/ReadVariableOpReadVariableOp(conv3d_25_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ц
conv3d_25/Conv3DConv3Dreshape_7/Reshape:output:0'conv3d_25/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_25/BiasAdd/ReadVariableOpReadVariableOp)conv3d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_25/BiasAddBiasAddconv3d_25/Conv3D:output:0(conv3d_25/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_25/ReluReluconv3d_25/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_20/Conv3D/ReadVariableOpReadVariableOp(conv3d_20_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ц
conv3d_20/Conv3DConv3Dreshape_7/Reshape:output:0'conv3d_20/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_20/BiasAdd/ReadVariableOpReadVariableOp)conv3d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_20/BiasAddBiasAddconv3d_20/Conv3D:output:0(conv3d_20/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_20/ReluReluconv3d_20/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ц
conv3d_15/Conv3DConv3Dreshape_7/Reshape:output:0'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/splitSplit(up_sampling3d_4/split/split_dim:output:0conv3d_25/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/concatConcatV2up_sampling3d_4/split:output:0up_sampling3d_4/split:output:0up_sampling3d_4/split:output:1up_sampling3d_4/split:output:1up_sampling3d_4/split:output:2up_sampling3d_4/split:output:2up_sampling3d_4/split:output:3up_sampling3d_4/split:output:3up_sampling3d_4/split:output:4up_sampling3d_4/split:output:4up_sampling3d_4/split:output:5up_sampling3d_4/split:output:5up_sampling3d_4/split:output:6up_sampling3d_4/split:output:6up_sampling3d_4/split:output:7up_sampling3d_4/split:output:7$up_sampling3d_4/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/split_1Split*up_sampling3d_4/split_1/split_dim:output:0up_sampling3d_4/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_4/concat_1ConcatV2 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:7 up_sampling3d_4/split_1:output:7&up_sampling3d_4/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_4/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_4/split_2Split*up_sampling3d_4/split_2/split_dim:output:0!up_sampling3d_4/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_4/concat_2ConcatV2 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:7 up_sampling3d_4/split_2:output:7&up_sampling3d_4/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0conv3d_20/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7$up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7&up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџc
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split_
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7&up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ_
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0conv3d_15/Relu:activations:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split[
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ы
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7"up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7$up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7$up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_26/Conv3D/ReadVariableOpReadVariableOp(conv3d_26_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_26/Conv3DConv3D!up_sampling3d_4/concat_2:output:0'conv3d_26/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_26/BiasAdd/ReadVariableOpReadVariableOp)conv3d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_26/BiasAddBiasAddconv3d_26/Conv3D:output:0(conv3d_26/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_26/ReluReluconv3d_26/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_21/Conv3D/ReadVariableOpReadVariableOp(conv3d_21_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_21/Conv3DConv3D!up_sampling3d_2/concat_2:output:0'conv3d_21/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_21/BiasAdd/ReadVariableOpReadVariableOp)conv3d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_21/BiasAddBiasAddconv3d_21/Conv3D:output:0(conv3d_21/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_21/ReluReluconv3d_21/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ы
conv3d_16/Conv3DConv3Dup_sampling3d/concat_2:output:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_27/Conv3D/ReadVariableOpReadVariableOp(conv3d_27_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_27/Conv3DConv3Dconv3d_26/Relu:activations:0'conv3d_27/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_27/BiasAdd/ReadVariableOpReadVariableOp)conv3d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_27/BiasAddBiasAddconv3d_27/Conv3D:output:0(conv3d_27/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_27/ReluReluconv3d_27/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_22/Conv3D/ReadVariableOpReadVariableOp(conv3d_22_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_22/Conv3DConv3Dconv3d_21/Relu:activations:0'conv3d_22/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_22/BiasAdd/ReadVariableOpReadVariableOp)conv3d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_22/BiasAddBiasAddconv3d_22/Conv3D:output:0(conv3d_22/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_22/ReluReluconv3d_22/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_17/Conv3DConv3Dconv3d_16/Relu:activations:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџp
conv3d_17/ReluReluconv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџa
up_sampling3d_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_5/splitSplit(up_sampling3d_5/split/split_dim:output:0conv3d_27/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :	
up_sampling3d_5/concatConcatV2up_sampling3d_5/split:output:0up_sampling3d_5/split:output:0up_sampling3d_5/split:output:1up_sampling3d_5/split:output:1up_sampling3d_5/split:output:2up_sampling3d_5/split:output:2up_sampling3d_5/split:output:3up_sampling3d_5/split:output:3up_sampling3d_5/split:output:4up_sampling3d_5/split:output:4up_sampling3d_5/split:output:5up_sampling3d_5/split:output:5up_sampling3d_5/split:output:6up_sampling3d_5/split:output:6up_sampling3d_5/split:output:7up_sampling3d_5/split:output:7up_sampling3d_5/split:output:8up_sampling3d_5/split:output:8up_sampling3d_5/split:output:9up_sampling3d_5/split:output:9up_sampling3d_5/split:output:10up_sampling3d_5/split:output:10up_sampling3d_5/split:output:11up_sampling3d_5/split:output:11up_sampling3d_5/split:output:12up_sampling3d_5/split:output:12up_sampling3d_5/split:output:13up_sampling3d_5/split:output:13up_sampling3d_5/split:output:14up_sampling3d_5/split:output:14up_sampling3d_5/split:output:15up_sampling3d_5/split:output:15$up_sampling3d_5/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ c
!up_sampling3d_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_5/split_1Split*up_sampling3d_5/split_1/split_dim:output:0up_sampling3d_5/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split_
up_sampling3d_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_5/concat_1ConcatV2 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:9 up_sampling3d_5/split_1:output:9!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:15!up_sampling3d_5/split_1:output:15&up_sampling3d_5/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  c
!up_sampling3d_5/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_5/split_2Split*up_sampling3d_5/split_2/split_dim:output:0!up_sampling3d_5/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_split_
up_sampling3d_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_5/concat_2ConcatV2 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:9 up_sampling3d_5/split_2:output:9!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:15!up_sampling3d_5/split_2:output:15&up_sampling3d_5/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   a
up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_3/splitSplit(up_sampling3d_3/split/split_dim:output:0conv3d_22/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :	
up_sampling3d_3/concatConcatV2up_sampling3d_3/split:output:0up_sampling3d_3/split:output:0up_sampling3d_3/split:output:1up_sampling3d_3/split:output:1up_sampling3d_3/split:output:2up_sampling3d_3/split:output:2up_sampling3d_3/split:output:3up_sampling3d_3/split:output:3up_sampling3d_3/split:output:4up_sampling3d_3/split:output:4up_sampling3d_3/split:output:5up_sampling3d_3/split:output:5up_sampling3d_3/split:output:6up_sampling3d_3/split:output:6up_sampling3d_3/split:output:7up_sampling3d_3/split:output:7up_sampling3d_3/split:output:8up_sampling3d_3/split:output:8up_sampling3d_3/split:output:9up_sampling3d_3/split:output:9up_sampling3d_3/split:output:10up_sampling3d_3/split:output:10up_sampling3d_3/split:output:11up_sampling3d_3/split:output:11up_sampling3d_3/split:output:12up_sampling3d_3/split:output:12up_sampling3d_3/split:output:13up_sampling3d_3/split:output:13up_sampling3d_3/split:output:14up_sampling3d_3/split:output:14up_sampling3d_3/split:output:15up_sampling3d_3/split:output:15$up_sampling3d_3/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ c
!up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_3/split_1Split*up_sampling3d_3/split_1/split_dim:output:0up_sampling3d_3/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split_
up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_3/concat_1ConcatV2 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:9 up_sampling3d_3/split_1:output:9!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:15&up_sampling3d_3/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  c
!up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_3/split_2Split*up_sampling3d_3/split_2/split_dim:output:0!up_sampling3d_3/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_split_
up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_3/concat_2ConcatV2 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:9 up_sampling3d_3/split_2:output:9!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:15&up_sampling3d_3/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   a
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0conv3d_17/Relu:activations:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split]
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :	
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15$up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ c
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split_
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15&up_sampling3d_1/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ  c
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*
_output_shapesѓ
№:џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  :џџџџџџџџџ  *
	num_split_
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :п	
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15&up_sampling3d_1/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_28/Conv3D/ReadVariableOpReadVariableOp(conv3d_28_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_28/Conv3DConv3D!up_sampling3d_5/concat_2:output:0'conv3d_28/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_28/BiasAdd/ReadVariableOpReadVariableOp)conv3d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_28/BiasAddBiasAddconv3d_28/Conv3D:output:0(conv3d_28/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_28/ReluReluconv3d_28/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_23/Conv3D/ReadVariableOpReadVariableOp(conv3d_23_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_23/Conv3DConv3D!up_sampling3d_3/concat_2:output:0'conv3d_23/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_23/BiasAdd/ReadVariableOpReadVariableOp)conv3d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_23/BiasAddBiasAddconv3d_23/Conv3D:output:0(conv3d_23/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_23/ReluReluconv3d_23/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_18/Conv3D/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Э
conv3d_18/Conv3DConv3D!up_sampling3d_1/concat_2:output:0'conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_18/BiasAdd/ReadVariableOpReadVariableOp)conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_18/BiasAddBiasAddconv3d_18/Conv3D:output:0(conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_18/ReluReluconv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_19/Conv3D/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_19/Conv3DConv3Dconv3d_18/Relu:activations:0'conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_19/BiasAdd/ReadVariableOpReadVariableOp)conv3d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_19/BiasAddBiasAddconv3d_19/Conv3D:output:0(conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_19/ReluReluconv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_24/Conv3D/ReadVariableOpReadVariableOp(conv3d_24_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_24/Conv3DConv3Dconv3d_23/Relu:activations:0'conv3d_24/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_24/BiasAdd/ReadVariableOpReadVariableOp)conv3d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_24/BiasAddBiasAddconv3d_24/Conv3D:output:0(conv3d_24/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_24/ReluReluconv3d_24/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_29/Conv3D/ReadVariableOpReadVariableOp(conv3d_29_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ш
conv3d_29/Conv3DConv3Dconv3d_28/Relu:activations:0'conv3d_29/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_29/BiasAdd/ReadVariableOpReadVariableOp)conv3d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_29/BiasAddBiasAddconv3d_29/Conv3D:output:0(conv3d_29/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   p
conv3d_29/ReluReluconv3d_29/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
	add_1/addAddV2conv3d_19/Relu:activations:0conv3d_24/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
add_1/add_1AddV2add_1/add:z:0conv3d_29/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ   
conv3d_30/Conv3D/ReadVariableOpReadVariableOp(conv3d_30_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0Л
conv3d_30/Conv3DConv3Dadd_1/add_1:z:0'conv3d_30/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	

 conv3d_30/BiasAdd/ReadVariableOpReadVariableOp)conv3d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv3d_30/BiasAddBiasAddconv3d_30/Conv3D:output:0(conv3d_30/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   u
IdentityIdentityconv3d_30/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp!^conv3d_18/BiasAdd/ReadVariableOp ^conv3d_18/Conv3D/ReadVariableOp!^conv3d_19/BiasAdd/ReadVariableOp ^conv3d_19/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp!^conv3d_20/BiasAdd/ReadVariableOp ^conv3d_20/Conv3D/ReadVariableOp!^conv3d_21/BiasAdd/ReadVariableOp ^conv3d_21/Conv3D/ReadVariableOp!^conv3d_22/BiasAdd/ReadVariableOp ^conv3d_22/Conv3D/ReadVariableOp!^conv3d_23/BiasAdd/ReadVariableOp ^conv3d_23/Conv3D/ReadVariableOp!^conv3d_24/BiasAdd/ReadVariableOp ^conv3d_24/Conv3D/ReadVariableOp!^conv3d_25/BiasAdd/ReadVariableOp ^conv3d_25/Conv3D/ReadVariableOp!^conv3d_26/BiasAdd/ReadVariableOp ^conv3d_26/Conv3D/ReadVariableOp!^conv3d_27/BiasAdd/ReadVariableOp ^conv3d_27/Conv3D/ReadVariableOp!^conv3d_28/BiasAdd/ReadVariableOp ^conv3d_28/Conv3D/ReadVariableOp!^conv3d_29/BiasAdd/ReadVariableOp ^conv3d_29/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp!^conv3d_30/BiasAdd/ReadVariableOp ^conv3d_30/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp2D
 conv3d_11/BiasAdd/ReadVariableOp conv3d_11/BiasAdd/ReadVariableOp2B
conv3d_11/Conv3D/ReadVariableOpconv3d_11/Conv3D/ReadVariableOp2D
 conv3d_12/BiasAdd/ReadVariableOp conv3d_12/BiasAdd/ReadVariableOp2B
conv3d_12/Conv3D/ReadVariableOpconv3d_12/Conv3D/ReadVariableOp2D
 conv3d_13/BiasAdd/ReadVariableOp conv3d_13/BiasAdd/ReadVariableOp2B
conv3d_13/Conv3D/ReadVariableOpconv3d_13/Conv3D/ReadVariableOp2D
 conv3d_14/BiasAdd/ReadVariableOp conv3d_14/BiasAdd/ReadVariableOp2B
conv3d_14/Conv3D/ReadVariableOpconv3d_14/Conv3D/ReadVariableOp2D
 conv3d_15/BiasAdd/ReadVariableOp conv3d_15/BiasAdd/ReadVariableOp2B
conv3d_15/Conv3D/ReadVariableOpconv3d_15/Conv3D/ReadVariableOp2D
 conv3d_16/BiasAdd/ReadVariableOp conv3d_16/BiasAdd/ReadVariableOp2B
conv3d_16/Conv3D/ReadVariableOpconv3d_16/Conv3D/ReadVariableOp2D
 conv3d_17/BiasAdd/ReadVariableOp conv3d_17/BiasAdd/ReadVariableOp2B
conv3d_17/Conv3D/ReadVariableOpconv3d_17/Conv3D/ReadVariableOp2D
 conv3d_18/BiasAdd/ReadVariableOp conv3d_18/BiasAdd/ReadVariableOp2B
conv3d_18/Conv3D/ReadVariableOpconv3d_18/Conv3D/ReadVariableOp2D
 conv3d_19/BiasAdd/ReadVariableOp conv3d_19/BiasAdd/ReadVariableOp2B
conv3d_19/Conv3D/ReadVariableOpconv3d_19/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2D
 conv3d_20/BiasAdd/ReadVariableOp conv3d_20/BiasAdd/ReadVariableOp2B
conv3d_20/Conv3D/ReadVariableOpconv3d_20/Conv3D/ReadVariableOp2D
 conv3d_21/BiasAdd/ReadVariableOp conv3d_21/BiasAdd/ReadVariableOp2B
conv3d_21/Conv3D/ReadVariableOpconv3d_21/Conv3D/ReadVariableOp2D
 conv3d_22/BiasAdd/ReadVariableOp conv3d_22/BiasAdd/ReadVariableOp2B
conv3d_22/Conv3D/ReadVariableOpconv3d_22/Conv3D/ReadVariableOp2D
 conv3d_23/BiasAdd/ReadVariableOp conv3d_23/BiasAdd/ReadVariableOp2B
conv3d_23/Conv3D/ReadVariableOpconv3d_23/Conv3D/ReadVariableOp2D
 conv3d_24/BiasAdd/ReadVariableOp conv3d_24/BiasAdd/ReadVariableOp2B
conv3d_24/Conv3D/ReadVariableOpconv3d_24/Conv3D/ReadVariableOp2D
 conv3d_25/BiasAdd/ReadVariableOp conv3d_25/BiasAdd/ReadVariableOp2B
conv3d_25/Conv3D/ReadVariableOpconv3d_25/Conv3D/ReadVariableOp2D
 conv3d_26/BiasAdd/ReadVariableOp conv3d_26/BiasAdd/ReadVariableOp2B
conv3d_26/Conv3D/ReadVariableOpconv3d_26/Conv3D/ReadVariableOp2D
 conv3d_27/BiasAdd/ReadVariableOp conv3d_27/BiasAdd/ReadVariableOp2B
conv3d_27/Conv3D/ReadVariableOpconv3d_27/Conv3D/ReadVariableOp2D
 conv3d_28/BiasAdd/ReadVariableOp conv3d_28/BiasAdd/ReadVariableOp2B
conv3d_28/Conv3D/ReadVariableOpconv3d_28/Conv3D/ReadVariableOp2D
 conv3d_29/BiasAdd/ReadVariableOp conv3d_29/BiasAdd/ReadVariableOp2B
conv3d_29/Conv3D/ReadVariableOpconv3d_29/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2D
 conv3d_30/BiasAdd/ReadVariableOp conv3d_30/BiasAdd/ReadVariableOp2B
conv3d_30/Conv3D/ReadVariableOpconv3d_30/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:Y U
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/2
р
L
0__inference_up_sampling3d_3_layer_call_fn_136450

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_132326l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_135916

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ё
L
0__inference_max_pooling3d_3_layer_call_fn_135926

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131513
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_23_layer_call_and_return_conditional_losses_136617

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
ь
g
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_132059

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_135961

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131826

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_135761

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs


E__inference_conv3d_14_layer_call_and_return_conditional_losses_136021

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџc
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
e
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_136175

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :я
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapesћ
ј:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_135781

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_135936

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І

D__inference_conv3d_2_layer_call_and_return_conditional_losses_131765

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
б
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_135586

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_10_layer_call_and_return_conditional_losses_131611

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
О
F
*__inference_reshape_7_layer_call_fn_136058

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_7_layer_call_and_return_conditional_losses_131930l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_26_layer_call_and_return_conditional_losses_136319

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131838

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_13_layer_call_fn_135890

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_131782{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
L
0__inference_max_pooling3d_5_layer_call_fn_135946

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131525
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_15_layer_call_and_return_conditional_losses_136093

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_24_layer_call_and_return_conditional_losses_136677

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs


E__inference_conv3d_14_layer_call_and_return_conditional_losses_131885

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџc
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_21_layer_call_and_return_conditional_losses_132128

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
І

D__inference_conv3d_8_layer_call_and_return_conditional_losses_135881

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_19_layer_call_and_return_conditional_losses_136657

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_29_layer_call_fn_136686

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_29_layer_call_and_return_conditional_losses_132487{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_28_layer_call_and_return_conditional_losses_132402

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Т!
Ќ
&__inference_model_layer_call_fn_134414
inputs_0
inputs_1
inputs_2%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:(

unknown_37:

unknown_38:(

unknown_39:

unknown_40:(

unknown_41:

unknown_42:(

unknown_43:

unknown_44:(

unknown_45:

unknown_46:(

unknown_47:

unknown_48:(

unknown_49:

unknown_50:(

unknown_51:

unknown_52:(

unknown_53:

unknown_54:(

unknown_55:

unknown_56:(

unknown_57:

unknown_58:(

unknown_59:

unknown_60:
identityЂStatefulPartitionedCallО	
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_133393{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/2
џ
 
'__inference_conv3d_layer_call_fn_135610

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_131645{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs


D__inference_conv3d_4_layer_call_and_return_conditional_losses_135981

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџc
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
L
0__inference_max_pooling3d_4_layer_call_fn_135766

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131489
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_18_layer_call_and_return_conditional_losses_132436

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_18_layer_call_and_return_conditional_losses_136597

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_23_layer_call_fn_136606

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_23_layer_call_and_return_conditional_losses_132419{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
р
L
0__inference_max_pooling3d_1_layer_call_fn_135911

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131838l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Т!
Ќ
&__inference_model_layer_call_fn_134283
inputs_0
inputs_1
inputs_2%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:(

unknown_37:

unknown_38:(

unknown_39:

unknown_40:(

unknown_41:

unknown_42:(

unknown_43:

unknown_44:(

unknown_45:

unknown_46:(

unknown_47:

unknown_48:(

unknown_49:

unknown_50:(

unknown_51:

unknown_52:(

unknown_53:

unknown_54:(

unknown_55:

unknown_56:(

unknown_57:

unknown_58:(

unknown_59:

unknown_60:
identityЂStatefulPartitionedCallО	
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_132520{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ   
"
_user_specified_name
inputs/2
ж
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131465

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ
^
$__inference_add_layer_call_fn_136028
inputs_0
inputs_1
inputs_2
identityг
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_131899l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:] Y
3
_output_shapes!
:џџџџџџџџџ
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:џџџџџџџџџ
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:џџџџџџџџџ
"
_user_specified_name
inputs/2
р
L
0__inference_max_pooling3d_5_layer_call_fn_135951

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131826l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч


E__inference_conv3d_30_layer_call_and_return_conditional_losses_136731

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_135756

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь
F
*__inference_reshape_1_layer_call_fn_135551

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_131571l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
ќ	
a
E__inference_reshape_6_layer_call_and_return_conditional_losses_131913

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_28_layer_call_and_return_conditional_losses_136637

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
р
L
0__inference_up_sampling3d_1_layer_call_fn_136384

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_132389l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_135776

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_135566

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
ы
y
A__inference_add_1_layer_call_and_return_conditional_losses_132501

inputs
inputs_1
inputs_2
identity\
addAddV2inputsinputs_1*
T0*3
_output_shapes!
:џџџџџџџџџ   _
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:џџџџџџџџџ   ]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs:[W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs:[W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
І

D__inference_conv3d_5_layer_call_and_return_conditional_losses_131628

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ђ
)__inference_conv3d_3_layer_call_fn_135850

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_131816{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
б
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_131588

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Q
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :У
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ   :W S
/
_output_shapes
:џџџџџџџџџ   
 
_user_specified_nameinputs
но
н
A__inference_model_layer_call_and_return_conditional_losses_133832	
u_vel	
v_vel	
w_vel.
conv3d_10_133660:
conv3d_10_133662:-
conv3d_5_133665:
conv3d_5_133667:+
conv3d_133670:
conv3d_133672:.
conv3d_11_133675:
conv3d_11_133677:-
conv3d_6_133680:
conv3d_6_133682:-
conv3d_1_133685:
conv3d_1_133687:.
conv3d_12_133693:
conv3d_12_133695:-
conv3d_7_133698:
conv3d_7_133700:-
conv3d_2_133703:
conv3d_2_133705:.
conv3d_13_133708:
conv3d_13_133710:-
conv3d_8_133713:
conv3d_8_133715:-
conv3d_3_133718:
conv3d_3_133720:-
conv3d_4_133726:
conv3d_4_133728:-
conv3d_9_133731:
conv3d_9_133733:.
conv3d_14_133736:
conv3d_14_133738:.
conv3d_25_133744:
conv3d_25_133746:.
conv3d_20_133749:
conv3d_20_133751:.
conv3d_15_133754:
conv3d_15_133756:.
conv3d_26_133762:
conv3d_26_133764:.
conv3d_21_133767:
conv3d_21_133769:.
conv3d_16_133772:
conv3d_16_133774:.
conv3d_27_133777:
conv3d_27_133779:.
conv3d_22_133782:
conv3d_22_133784:.
conv3d_17_133787:
conv3d_17_133789:.
conv3d_28_133795:
conv3d_28_133797:.
conv3d_23_133800:
conv3d_23_133802:.
conv3d_18_133805:
conv3d_18_133807:.
conv3d_19_133810:
conv3d_19_133812:.
conv3d_24_133815:
conv3d_24_133817:.
conv3d_29_133820:
conv3d_29_133822:.
conv3d_30_133826:
conv3d_30_133828:
identityЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ!conv3d_10/StatefulPartitionedCallЂ!conv3d_11/StatefulPartitionedCallЂ!conv3d_12/StatefulPartitionedCallЂ!conv3d_13/StatefulPartitionedCallЂ!conv3d_14/StatefulPartitionedCallЂ!conv3d_15/StatefulPartitionedCallЂ!conv3d_16/StatefulPartitionedCallЂ!conv3d_17/StatefulPartitionedCallЂ!conv3d_18/StatefulPartitionedCallЂ!conv3d_19/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂ!conv3d_20/StatefulPartitionedCallЂ!conv3d_21/StatefulPartitionedCallЂ!conv3d_22/StatefulPartitionedCallЂ!conv3d_23/StatefulPartitionedCallЂ!conv3d_24/StatefulPartitionedCallЂ!conv3d_25/StatefulPartitionedCallЂ!conv3d_26/StatefulPartitionedCallЂ!conv3d_27/StatefulPartitionedCallЂ!conv3d_28/StatefulPartitionedCallЂ!conv3d_29/StatefulPartitionedCallЂ conv3d_3/StatefulPartitionedCallЂ!conv3d_30/StatefulPartitionedCallЂ conv3d_4/StatefulPartitionedCallЂ conv3d_5/StatefulPartitionedCallЂ conv3d_6/StatefulPartitionedCallЂ conv3d_7/StatefulPartitionedCallЂ conv3d_8/StatefulPartitionedCallЂ conv3d_9/StatefulPartitionedCallЦ
reshape/PartitionedCallPartitionedCallu_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_131554Ъ
reshape_1/PartitionedCallPartitionedCallv_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_131571Ъ
reshape_2/PartitionedCallPartitionedCallw_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_131588Г
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_131598Ѓ
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_10_133660conv3d_10_133662*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_131611
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_5_133665conv3d_5_133667*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_131628
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_133670conv3d_133672*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_131645Љ
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_133675conv3d_11_133677*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_131662Є
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_133680conv3d_6_133682*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_131679Ђ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_133685conv3d_1_133687*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_131696ћ
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131706њ
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_131712і
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_131718Ї
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_133693conv3d_12_133695*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_131731Ѓ
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_133698conv3d_7_133700*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_131748Ё
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_133703conv3d_2_133705*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_131765Љ
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_133708conv3d_13_133710*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_131782Є
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_133713conv3d_8_133715*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_131799Є
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_133718conv3d_3_133720*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_131816ћ
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_131826њ
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_131832њ
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131838Ѓ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_133726conv3d_4_133728*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_131851Ѓ
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_133731conv3d_9_133733*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_131868Ї
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_133736conv3d_14_133738*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_131885Л
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_131899ж
reshape_6/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_131913ч
reshape_7/PartitionedCallPartitionedCall"reshape_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_7_layer_call_and_return_conditional_losses_131930Ё
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_25_133744conv3d_25_133746*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_25_layer_call_and_return_conditional_losses_131943Ё
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_20_133749conv3d_20_133751*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_20_layer_call_and_return_conditional_losses_131960Ё
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv3d_15_133754conv3d_15_133756*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_131977ћ
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_132020ћ
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_132059ї
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_132098Ї
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_133762conv3d_26_133764*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_26_layer_call_and_return_conditional_losses_132111Ї
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_133767conv3d_21_133769*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_21_layer_call_and_return_conditional_losses_132128Ѕ
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_133772conv3d_16_133774*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_132145Љ
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_133777conv3d_27_133779*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_27_layer_call_and_return_conditional_losses_132162Љ
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_133782conv3d_22_133784*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_22_layer_call_and_return_conditional_losses_132179Љ
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_133787conv3d_17_133789*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_132196ћ
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_132263ћ
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_132326ћ
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_132389Ї
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_133795conv3d_28_133797*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_28_layer_call_and_return_conditional_losses_132402Ї
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_133800conv3d_23_133802*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_23_layer_call_and_return_conditional_losses_132419Ї
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_133805conv3d_18_133807*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_132436Љ
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_133810conv3d_19_133812*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_132453Љ
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_133815conv3d_24_133817*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_24_layer_call_and_return_conditional_losses_132470Љ
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_133820conv3d_29_133822*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_29_layer_call_and_return_conditional_losses_132487С
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_132501
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_133826conv3d_30_133828*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_30_layer_call_and_return_conditional_losses_132513
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Э:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2F
!conv3d_20/StatefulPartitionedCall!conv3d_20/StatefulPartitionedCall2F
!conv3d_21/StatefulPartitionedCall!conv3d_21/StatefulPartitionedCall2F
!conv3d_22/StatefulPartitionedCall!conv3d_22/StatefulPartitionedCall2F
!conv3d_23/StatefulPartitionedCall!conv3d_23/StatefulPartitionedCall2F
!conv3d_24/StatefulPartitionedCall!conv3d_24/StatefulPartitionedCall2F
!conv3d_25/StatefulPartitionedCall!conv3d_25/StatefulPartitionedCall2F
!conv3d_26/StatefulPartitionedCall!conv3d_26/StatefulPartitionedCall2F
!conv3d_27/StatefulPartitionedCall!conv3d_27/StatefulPartitionedCall2F
!conv3d_28/StatefulPartitionedCall!conv3d_28/StatefulPartitionedCall2F
!conv3d_29/StatefulPartitionedCall!conv3d_29/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2F
!conv3d_30/StatefulPartitionedCall!conv3d_30/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ   

_user_specified_nameu_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namev_vel:VR
/
_output_shapes
:џџџџџџџџџ   

_user_specified_namew_vel
р
L
0__inference_max_pooling3d_4_layer_call_fn_135771

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_131706l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ   :[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_28_layer_call_fn_136626

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_28_layer_call_and_return_conditional_losses_132402{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_26_layer_call_and_return_conditional_losses_132111

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

f
,__inference_concatenate_layer_call_fn_135593
inputs_0
inputs_1
inputs_2
identityл
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_131598l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџ   :џџџџџџџџџ   :џџџџџџџџџ   :] Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:џџџџџџџџџ   
"
_user_specified_name
inputs/2
І

D__inference_conv3d_7_layer_call_and_return_conditional_losses_131748

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_22_layer_call_and_return_conditional_losses_132179

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_18_layer_call_fn_136586

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_132436{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_24_layer_call_fn_136666

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_24_layer_call_and_return_conditional_losses_132470{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
и
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_131501

inputs
identityН
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ч
g
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_135941

inputs
identity
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs

Ѓ
*__inference_conv3d_30_layer_call_fn_136721

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ   *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_30_layer_call_and_return_conditional_losses_132513{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_23_layer_call_and_return_conditional_losses_132419

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs
Ї

E__inference_conv3d_16_layer_call_and_return_conditional_losses_136279

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

E__inference_conv3d_11_layer_call_and_return_conditional_losses_131662

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:џџџџџџџџџ   
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
?
u_vel6
serving_default_u_vel:0џџџџџџџџџ   
?
v_vel6
serving_default_v_vel:0џџџџџџџџџ   
?
w_vel6
serving_default_w_vel:0џџџџџџџџџ   I
	conv3d_30<
StatefulPartitionedCall:0џџџџџџџџџ   tensorflow/serving/predict:ЌЏ

layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-15
 layer-31
!layer_with_weights-16
!layer-32
"layer_with_weights-17
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-18
&layer-37
'layer_with_weights-19
'layer-38
(layer_with_weights-20
(layer-39
)layer_with_weights-21
)layer-40
*layer_with_weights-22
*layer-41
+layer_with_weights-23
+layer-42
,layer-43
-layer-44
.layer-45
/layer_with_weights-24
/layer-46
0layer_with_weights-25
0layer-47
1layer_with_weights-26
1layer-48
2layer_with_weights-27
2layer-49
3layer_with_weights-28
3layer-50
4layer_with_weights-29
4layer-51
5layer-52
6layer_with_weights-30
6layer-53
7	optimizer
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<
signatures
б__call__
+в&call_and_return_all_conditional_losses
г_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ї
=	variables
>trainable_variables
?regularization_losses
@	keras_api
д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
и__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
к__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
м__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
о__call__
+п&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
р__call__
+с&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
ф__call__
+х&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
ц__call__
+ч&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
Р

}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api
ю__call__
+я&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
№__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
ђ__call__
+ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
є__call__
+ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
і__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
 	keras_api
ј__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
њ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
ќ__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
ў__call__
+џ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
­kernel
	Ўbias
Џ	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Гkernel
	Дbias
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Йkernel
	Кbias
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Ыkernel
	Ьbias
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
бkernel
	вbias
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
зkernel
	иbias
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
щkernel
	ъbias
ы	variables
ьtrainable_variables
эregularization_losses
ю	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
яkernel
	№bias
ё	variables
ђtrainable_variables
ѓregularization_losses
є	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
ѕkernel
	іbias
ї	variables
јtrainable_variables
љregularization_losses
њ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
ћkernel
	ќbias
§	variables
ўtrainable_variables
џregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
Є__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	 bias
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Ѕkernel
	Іbias
Ї	variables
Јtrainable_variables
Љregularization_losses
Њ	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Ћkernel
	Ќbias
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Бkernel
	Вbias
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Зkernel
	Иbias
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Сkernel
	Тbias
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer

	Чiter
Шbeta_1
Щbeta_2

Ъdecay
Ыlearning_rateMmеNmжSmзTmиYmйZmк_mл`mмemнfmоkmпlmр}mс~mт	mу	mф	mх	mц	mч	mш	mщ	mъ	mы	mь	­mэ	Ўmю	Гmя	Дm№	Йmё	Кmђ	Ыmѓ	Ьmє	бmѕ	вmі	зmї	иmј	щmљ	ъmњ	яmћ	№mќ	ѕm§	іmў	ћmџ	ќm	m	m	m	m	m	m	m	 m	Ѕm	Іm	Ћm	Ќm	Бm	Вm	Зm	Иm	Сm	ТmMvNvSvTvYvZv_v`vevfvkvlv}v~v 	vЁ	vЂ	vЃ	vЄ	vЅ	vІ	vЇ	vЈ	vЉ	vЊ	­vЋ	ЎvЌ	Гv­	ДvЎ	ЙvЏ	КvА	ЫvБ	ЬvВ	бvГ	вvД	зvЕ	иvЖ	щvЗ	ъvИ	яvЙ	№vК	ѕvЛ	іvМ	ћvН	ќvО	vП	vР	vС	vТ	vУ	vФ	vХ	 vЦ	ЅvЧ	ІvШ	ЋvЩ	ЌvЪ	БvЫ	ВvЬ	ЗvЭ	ИvЮ	СvЯ	Тvа"
	optimizer
Ж
M0
N1
S2
T3
Y4
Z5
_6
`7
e8
f9
k10
l11
}12
~13
14
15
16
17
18
19
20
21
22
23
­24
Ў25
Г26
Д27
Й28
К29
Ы30
Ь31
б32
в33
з34
и35
щ36
ъ37
я38
№39
ѕ40
і41
ћ42
ќ43
44
45
46
47
48
49
50
 51
Ѕ52
І53
Ћ54
Ќ55
Б56
В57
З58
И59
С60
Т61"
trackable_list_wrapper
Ж
M0
N1
S2
T3
Y4
Z5
_6
`7
e8
f9
k10
l11
}12
~13
14
15
16
17
18
19
20
21
22
23
­24
Ў25
Г26
Д27
Й28
К29
Ы30
Ь31
б32
в33
з34
и35
щ36
ъ37
я38
№39
ѕ40
і41
ћ42
ќ43
44
45
46
47
48
49
50
 51
Ѕ52
І53
Ћ54
Ќ55
Б56
В57
З58
И59
С60
Т61"
trackable_list_wrapper
 "
trackable_list_wrapper
г
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
8	variables
9trainable_variables
:regularization_losses
б__call__
г_default_save_signature
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
-
Кserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
=	variables
>trainable_variables
?regularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
+:)2conv3d/kernel
:2conv3d/bias
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
Е
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_10/kernel
:2conv3d_10/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
[	variables
\trainable_variables
]regularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_1/kernel
:2conv3d_1/bias
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
a	variables
btrainable_variables
cregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_6/kernel
:2conv3d_6/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
g	variables
htrainable_variables
iregularization_losses
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_11/kernel
:2conv3d_11/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ўnon_trainable_variables
џlayers
metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
ц__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
q	variables
rtrainable_variables
sregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
y	variables
ztrainable_variables
{regularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_2/kernel
:2conv3d_2/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
З
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
ю__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_7/kernel
:2conv3d_7/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_12/kernel
:2conv3d_12/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_3/kernel
:2conv3d_3/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
	variables
trainable_variables
regularization_losses
є__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_8/kernel
:2conv3d_8/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
	variables
trainable_variables
regularization_losses
і__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_13/kernel
:2conv3d_13/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
	variables
trainable_variables
regularization_losses
ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
ќ__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_4/kernel
:2conv3d_4/bias
0
­0
Ў1"
trackable_list_wrapper
0
­0
Ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
Џ	variables
Аtrainable_variables
Бregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_9/kernel
:2conv3d_9/bias
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_14/kernel
:2conv3d_14/bias
0
Й0
К1"
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_15/kernel
:2conv3d_15/bias
0
Ы0
Ь1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_20/kernel
:2conv3d_20/bias
0
б0
в1"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
г	variables
дtrainable_variables
еregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_25/kernel
:2conv3d_25/bias
0
з0
и1"
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
н	variables
оtrainable_variables
пregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ёnon_trainable_variables
ђlayers
ѓmetrics
 єlayer_regularization_losses
ѕlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_16/kernel
:2conv3d_16/bias
0
щ0
ъ1"
trackable_list_wrapper
0
щ0
ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
ы	variables
ьtrainable_variables
эregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_21/kernel
:2conv3d_21/bias
0
я0
№1"
trackable_list_wrapper
0
я0
№1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ё	variables
ђtrainable_variables
ѓregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_26/kernel
:2conv3d_26/bias
0
ѕ0
і1"
trackable_list_wrapper
0
ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ї	variables
јtrainable_variables
љregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_17/kernel
:2conv3d_17/bias
0
ћ0
ќ1"
trackable_list_wrapper
0
ћ0
ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
ўtrainable_variables
џregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_22/kernel
:2conv3d_22/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_27/kernel
:2conv3d_27/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
Є__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѓnon_trainable_variables
Єlayers
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
	variables
trainable_variables
regularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_18/kernel
:2conv3d_18/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
	variables
trainable_variables
regularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_23/kernel
:2conv3d_23/bias
0
0
 1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_28/kernel
:2conv3d_28/bias
0
Ѕ0
І1"
trackable_list_wrapper
0
Ѕ0
І1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Ї	variables
Јtrainable_variables
Љregularization_losses
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_19/kernel
:2conv3d_19/bias
0
Ћ0
Ќ1"
trackable_list_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_24/kernel
:2conv3d_24/bias
0
Б0
В1"
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_29/kernel
:2conv3d_29/bias
0
З0
И1"
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_30/kernel
:2conv3d_30/bias
0
С0
Т1"
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653"
trackable_list_wrapper
(
а0"
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
R

бtotal

вcount
г	variables
д	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
б0
в1"
trackable_list_wrapper
.
г	variables"
_generic_user_object
0:.2Adam/conv3d/kernel/m
:2Adam/conv3d/bias/m
2:02Adam/conv3d_5/kernel/m
 :2Adam/conv3d_5/bias/m
3:12Adam/conv3d_10/kernel/m
!:2Adam/conv3d_10/bias/m
2:02Adam/conv3d_1/kernel/m
 :2Adam/conv3d_1/bias/m
2:02Adam/conv3d_6/kernel/m
 :2Adam/conv3d_6/bias/m
3:12Adam/conv3d_11/kernel/m
!:2Adam/conv3d_11/bias/m
2:02Adam/conv3d_2/kernel/m
 :2Adam/conv3d_2/bias/m
2:02Adam/conv3d_7/kernel/m
 :2Adam/conv3d_7/bias/m
3:12Adam/conv3d_12/kernel/m
!:2Adam/conv3d_12/bias/m
2:02Adam/conv3d_3/kernel/m
 :2Adam/conv3d_3/bias/m
2:02Adam/conv3d_8/kernel/m
 :2Adam/conv3d_8/bias/m
3:12Adam/conv3d_13/kernel/m
!:2Adam/conv3d_13/bias/m
2:02Adam/conv3d_4/kernel/m
 :2Adam/conv3d_4/bias/m
2:02Adam/conv3d_9/kernel/m
 :2Adam/conv3d_9/bias/m
3:12Adam/conv3d_14/kernel/m
!:2Adam/conv3d_14/bias/m
3:12Adam/conv3d_15/kernel/m
!:2Adam/conv3d_15/bias/m
3:12Adam/conv3d_20/kernel/m
!:2Adam/conv3d_20/bias/m
3:12Adam/conv3d_25/kernel/m
!:2Adam/conv3d_25/bias/m
3:12Adam/conv3d_16/kernel/m
!:2Adam/conv3d_16/bias/m
3:12Adam/conv3d_21/kernel/m
!:2Adam/conv3d_21/bias/m
3:12Adam/conv3d_26/kernel/m
!:2Adam/conv3d_26/bias/m
3:12Adam/conv3d_17/kernel/m
!:2Adam/conv3d_17/bias/m
3:12Adam/conv3d_22/kernel/m
!:2Adam/conv3d_22/bias/m
3:12Adam/conv3d_27/kernel/m
!:2Adam/conv3d_27/bias/m
3:12Adam/conv3d_18/kernel/m
!:2Adam/conv3d_18/bias/m
3:12Adam/conv3d_23/kernel/m
!:2Adam/conv3d_23/bias/m
3:12Adam/conv3d_28/kernel/m
!:2Adam/conv3d_28/bias/m
3:12Adam/conv3d_19/kernel/m
!:2Adam/conv3d_19/bias/m
3:12Adam/conv3d_24/kernel/m
!:2Adam/conv3d_24/bias/m
3:12Adam/conv3d_29/kernel/m
!:2Adam/conv3d_29/bias/m
3:12Adam/conv3d_30/kernel/m
!:2Adam/conv3d_30/bias/m
0:.2Adam/conv3d/kernel/v
:2Adam/conv3d/bias/v
2:02Adam/conv3d_5/kernel/v
 :2Adam/conv3d_5/bias/v
3:12Adam/conv3d_10/kernel/v
!:2Adam/conv3d_10/bias/v
2:02Adam/conv3d_1/kernel/v
 :2Adam/conv3d_1/bias/v
2:02Adam/conv3d_6/kernel/v
 :2Adam/conv3d_6/bias/v
3:12Adam/conv3d_11/kernel/v
!:2Adam/conv3d_11/bias/v
2:02Adam/conv3d_2/kernel/v
 :2Adam/conv3d_2/bias/v
2:02Adam/conv3d_7/kernel/v
 :2Adam/conv3d_7/bias/v
3:12Adam/conv3d_12/kernel/v
!:2Adam/conv3d_12/bias/v
2:02Adam/conv3d_3/kernel/v
 :2Adam/conv3d_3/bias/v
2:02Adam/conv3d_8/kernel/v
 :2Adam/conv3d_8/bias/v
3:12Adam/conv3d_13/kernel/v
!:2Adam/conv3d_13/bias/v
2:02Adam/conv3d_4/kernel/v
 :2Adam/conv3d_4/bias/v
2:02Adam/conv3d_9/kernel/v
 :2Adam/conv3d_9/bias/v
3:12Adam/conv3d_14/kernel/v
!:2Adam/conv3d_14/bias/v
3:12Adam/conv3d_15/kernel/v
!:2Adam/conv3d_15/bias/v
3:12Adam/conv3d_20/kernel/v
!:2Adam/conv3d_20/bias/v
3:12Adam/conv3d_25/kernel/v
!:2Adam/conv3d_25/bias/v
3:12Adam/conv3d_16/kernel/v
!:2Adam/conv3d_16/bias/v
3:12Adam/conv3d_21/kernel/v
!:2Adam/conv3d_21/bias/v
3:12Adam/conv3d_26/kernel/v
!:2Adam/conv3d_26/bias/v
3:12Adam/conv3d_17/kernel/v
!:2Adam/conv3d_17/bias/v
3:12Adam/conv3d_22/kernel/v
!:2Adam/conv3d_22/bias/v
3:12Adam/conv3d_27/kernel/v
!:2Adam/conv3d_27/bias/v
3:12Adam/conv3d_18/kernel/v
!:2Adam/conv3d_18/bias/v
3:12Adam/conv3d_23/kernel/v
!:2Adam/conv3d_23/bias/v
3:12Adam/conv3d_28/kernel/v
!:2Adam/conv3d_28/bias/v
3:12Adam/conv3d_19/kernel/v
!:2Adam/conv3d_19/bias/v
3:12Adam/conv3d_24/kernel/v
!:2Adam/conv3d_24/bias/v
3:12Adam/conv3d_29/kernel/v
!:2Adam/conv3d_29/bias/v
3:12Adam/conv3d_30/kernel/v
!:2Adam/conv3d_30/bias/v
ц2у
&__inference_model_layer_call_fn_132647
&__inference_model_layer_call_fn_134283
&__inference_model_layer_call_fn_134414
&__inference_model_layer_call_fn_133651Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
A__inference_model_layer_call_and_return_conditional_losses_134970
A__inference_model_layer_call_and_return_conditional_losses_135526
A__inference_model_layer_call_and_return_conditional_losses_133832
A__inference_model_layer_call_and_return_conditional_losses_134013Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
иBе
!__inference__wrapped_model_131456u_velv_velw_vel"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_reshape_layer_call_fn_135531Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_reshape_layer_call_and_return_conditional_losses_135546Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_reshape_1_layer_call_fn_135551Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_reshape_1_layer_call_and_return_conditional_losses_135566Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_reshape_2_layer_call_fn_135571Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_reshape_2_layer_call_and_return_conditional_losses_135586Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_concatenate_layer_call_fn_135593Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_concatenate_layer_call_and_return_conditional_losses_135601Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_conv3d_layer_call_fn_135610Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_conv3d_layer_call_and_return_conditional_losses_135621Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_5_layer_call_fn_135630Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_5_layer_call_and_return_conditional_losses_135641Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_10_layer_call_fn_135650Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_10_layer_call_and_return_conditional_losses_135661Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_1_layer_call_fn_135670Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_1_layer_call_and_return_conditional_losses_135681Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_6_layer_call_fn_135690Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_6_layer_call_and_return_conditional_losses_135701Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_11_layer_call_fn_135710Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_11_layer_call_and_return_conditional_losses_135721Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
.__inference_max_pooling3d_layer_call_fn_135726
.__inference_max_pooling3d_layer_call_fn_135731Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О2Л
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_135736
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_135741Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_max_pooling3d_2_layer_call_fn_135746
0__inference_max_pooling3d_2_layer_call_fn_135751Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_135756
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_135761Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_max_pooling3d_4_layer_call_fn_135766
0__inference_max_pooling3d_4_layer_call_fn_135771Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_135776
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_135781Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_2_layer_call_fn_135790Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_2_layer_call_and_return_conditional_losses_135801Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_7_layer_call_fn_135810Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_7_layer_call_and_return_conditional_losses_135821Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_12_layer_call_fn_135830Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_12_layer_call_and_return_conditional_losses_135841Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_3_layer_call_fn_135850Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_3_layer_call_and_return_conditional_losses_135861Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_8_layer_call_fn_135870Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_8_layer_call_and_return_conditional_losses_135881Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_13_layer_call_fn_135890Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_13_layer_call_and_return_conditional_losses_135901Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_max_pooling3d_1_layer_call_fn_135906
0__inference_max_pooling3d_1_layer_call_fn_135911Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_135916
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_135921Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_max_pooling3d_3_layer_call_fn_135926
0__inference_max_pooling3d_3_layer_call_fn_135931Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_135936
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_135941Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_max_pooling3d_5_layer_call_fn_135946
0__inference_max_pooling3d_5_layer_call_fn_135951Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_135956
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_135961Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_4_layer_call_fn_135970Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_4_layer_call_and_return_conditional_losses_135981Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv3d_9_layer_call_fn_135990Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv3d_9_layer_call_and_return_conditional_losses_136001Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_14_layer_call_fn_136010Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_14_layer_call_and_return_conditional_losses_136021Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ю2Ы
$__inference_add_layer_call_fn_136028Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_add_layer_call_and_return_conditional_losses_136036Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_reshape_6_layer_call_fn_136041Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_reshape_6_layer_call_and_return_conditional_losses_136053Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_reshape_7_layer_call_fn_136058Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_reshape_7_layer_call_and_return_conditional_losses_136073Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_15_layer_call_fn_136082Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_15_layer_call_and_return_conditional_losses_136093Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_20_layer_call_fn_136102Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_20_layer_call_and_return_conditional_losses_136113Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_25_layer_call_fn_136122Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_25_layer_call_and_return_conditional_losses_136133Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_up_sampling3d_layer_call_fn_136138Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_136175Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_up_sampling3d_2_layer_call_fn_136180Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_136217Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_up_sampling3d_4_layer_call_fn_136222Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_136259Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_16_layer_call_fn_136268Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_16_layer_call_and_return_conditional_losses_136279Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_21_layer_call_fn_136288Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_21_layer_call_and_return_conditional_losses_136299Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_26_layer_call_fn_136308Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_26_layer_call_and_return_conditional_losses_136319Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_17_layer_call_fn_136328Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_17_layer_call_and_return_conditional_losses_136339Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_22_layer_call_fn_136348Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_22_layer_call_and_return_conditional_losses_136359Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_27_layer_call_fn_136368Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_27_layer_call_and_return_conditional_losses_136379Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_up_sampling3d_1_layer_call_fn_136384Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_136445Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_up_sampling3d_3_layer_call_fn_136450Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_136511Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_up_sampling3d_5_layer_call_fn_136516Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_136577Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_18_layer_call_fn_136586Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_18_layer_call_and_return_conditional_losses_136597Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_23_layer_call_fn_136606Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_23_layer_call_and_return_conditional_losses_136617Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_28_layer_call_fn_136626Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_28_layer_call_and_return_conditional_losses_136637Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_19_layer_call_fn_136646Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_19_layer_call_and_return_conditional_losses_136657Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_24_layer_call_fn_136666Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_24_layer_call_and_return_conditional_losses_136677Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_29_layer_call_fn_136686Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_29_layer_call_and_return_conditional_losses_136697Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_add_1_layer_call_fn_136704Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_add_1_layer_call_and_return_conditional_losses_136712Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv3d_30_layer_call_fn_136721Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv3d_30_layer_call_and_return_conditional_losses_136731Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
еBв
$__inference_signature_wrapper_134152u_velv_velw_vel"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ь
!__inference__wrapped_model_131456ЦnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЂ
Ђ
~{
'$
u_velџџџџџџџџџ   
'$
v_velџџџџџџџџџ   
'$
w_velџџџџџџџџџ   
Њ "AЊ>
<
	conv3d_30/,
	conv3d_30џџџџџџџџџ   Ѓ
A__inference_add_1_layer_call_and_return_conditional_losses_136712нЇЂЃ
Ђ

.+
inputs/0џџџџџџџџџ   
.+
inputs/1џџџџџџџџџ   
.+
inputs/2џџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 ћ
&__inference_add_1_layer_call_fn_136704аЇЂЃ
Ђ

.+
inputs/0џџџџџџџџџ   
.+
inputs/1џџџџџџџџџ   
.+
inputs/2џџџџџџџџџ   
Њ "$!џџџџџџџџџ   Ё
?__inference_add_layer_call_and_return_conditional_losses_136036нЇЂЃ
Ђ

.+
inputs/0џџџџџџџџџ
.+
inputs/1џџџџџџџџџ
.+
inputs/2џџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 љ
$__inference_add_layer_call_fn_136028аЇЂЃ
Ђ

.+
inputs/0џџџџџџџџџ
.+
inputs/1џџџџџџџџџ
.+
inputs/2џџџџџџџџџ
Њ "$!џџџџџџџџџЉ
G__inference_concatenate_layer_call_and_return_conditional_losses_135601нЇЂЃ
Ђ

.+
inputs/0џџџџџџџџџ   
.+
inputs/1џџџџџџџџџ   
.+
inputs/2џџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
,__inference_concatenate_layer_call_fn_135593аЇЂЃ
Ђ

.+
inputs/0џџџџџџџџџ   
.+
inputs/1џџџџџџџџџ   
.+
inputs/2џџџџџџџџџ   
Њ "$!џџџџџџџџџ   Н
E__inference_conv3d_10_layer_call_and_return_conditional_losses_135661tYZ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_10_layer_call_fn_135650gYZ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   Н
E__inference_conv3d_11_layer_call_and_return_conditional_losses_135721tkl;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_11_layer_call_fn_135710gkl;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   П
E__inference_conv3d_12_layer_call_and_return_conditional_losses_135841v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_12_layer_call_fn_135830i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_13_layer_call_and_return_conditional_losses_135901v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_13_layer_call_fn_135890i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_14_layer_call_and_return_conditional_losses_136021vЙК;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_14_layer_call_fn_136010iЙК;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_15_layer_call_and_return_conditional_losses_136093vЫЬ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_15_layer_call_fn_136082iЫЬ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_16_layer_call_and_return_conditional_losses_136279vщъ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_16_layer_call_fn_136268iщъ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_17_layer_call_and_return_conditional_losses_136339vћќ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_17_layer_call_fn_136328iћќ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_18_layer_call_and_return_conditional_losses_136597v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_18_layer_call_fn_136586i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   П
E__inference_conv3d_19_layer_call_and_return_conditional_losses_136657vЋЌ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_19_layer_call_fn_136646iЋЌ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   М
D__inference_conv3d_1_layer_call_and_return_conditional_losses_135681t_`;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
)__inference_conv3d_1_layer_call_fn_135670g_`;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   П
E__inference_conv3d_20_layer_call_and_return_conditional_losses_136113vбв;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_20_layer_call_fn_136102iбв;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_21_layer_call_and_return_conditional_losses_136299vя№;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_21_layer_call_fn_136288iя№;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_22_layer_call_and_return_conditional_losses_136359v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_22_layer_call_fn_136348i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_23_layer_call_and_return_conditional_losses_136617v ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_23_layer_call_fn_136606i ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   П
E__inference_conv3d_24_layer_call_and_return_conditional_losses_136677vБВ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_24_layer_call_fn_136666iБВ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   П
E__inference_conv3d_25_layer_call_and_return_conditional_losses_136133vзи;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_25_layer_call_fn_136122iзи;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_26_layer_call_and_return_conditional_losses_136319vѕі;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_26_layer_call_fn_136308iѕі;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_27_layer_call_and_return_conditional_losses_136379v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_conv3d_27_layer_call_fn_136368i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_28_layer_call_and_return_conditional_losses_136637vЅІ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_28_layer_call_fn_136626iЅІ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   П
E__inference_conv3d_29_layer_call_and_return_conditional_losses_136697vЗИ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_29_layer_call_fn_136686iЗИ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   М
D__inference_conv3d_2_layer_call_and_return_conditional_losses_135801t}~;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
)__inference_conv3d_2_layer_call_fn_135790g}~;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
E__inference_conv3d_30_layer_call_and_return_conditional_losses_136731vСТ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_conv3d_30_layer_call_fn_136721iСТ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   О
D__inference_conv3d_3_layer_call_and_return_conditional_losses_135861v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
)__inference_conv3d_3_layer_call_fn_135850i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџО
D__inference_conv3d_4_layer_call_and_return_conditional_losses_135981v­Ў;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
)__inference_conv3d_4_layer_call_fn_135970i­Ў;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџМ
D__inference_conv3d_5_layer_call_and_return_conditional_losses_135641tST;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
)__inference_conv3d_5_layer_call_fn_135630gST;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   М
D__inference_conv3d_6_layer_call_and_return_conditional_losses_135701tef;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
)__inference_conv3d_6_layer_call_fn_135690gef;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   О
D__inference_conv3d_7_layer_call_and_return_conditional_losses_135821v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
)__inference_conv3d_7_layer_call_fn_135810i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџО
D__inference_conv3d_8_layer_call_and_return_conditional_losses_135881v;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
)__inference_conv3d_8_layer_call_fn_135870i;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџО
D__inference_conv3d_9_layer_call_and_return_conditional_losses_136001vГД;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
)__inference_conv3d_9_layer_call_fn_135990iГД;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџК
B__inference_conv3d_layer_call_and_return_conditional_losses_135621tMN;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
'__inference_conv3d_layer_call_fn_135610gMN;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_135916И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_135921p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 р
0__inference_max_pooling3d_1_layer_call_fn_135906Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
0__inference_max_pooling3d_1_layer_call_fn_135911c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_135756И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_135761p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ
 р
0__inference_max_pooling3d_2_layer_call_fn_135746Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
0__inference_max_pooling3d_2_layer_call_fn_135751c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_135936И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_135941p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 р
0__inference_max_pooling3d_3_layer_call_fn_135926Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
0__inference_max_pooling3d_3_layer_call_fn_135931c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_135776И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
K__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_135781p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ
 р
0__inference_max_pooling3d_4_layer_call_fn_135766Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
0__inference_max_pooling3d_4_layer_call_fn_135771c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_135956И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
K__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_135961p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 р
0__inference_max_pooling3d_5_layer_call_fn_135946Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
0__inference_max_pooling3d_5_layer_call_fn_135951c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_135736И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_135741p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ
 о
.__inference_max_pooling3d_layer_call_fn_135726Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
.__inference_max_pooling3d_layer_call_fn_135731c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ
A__inference_model_layer_call_and_return_conditional_losses_133832ОnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЂ
Ђ
~{
'$
u_velџџџџџџџџџ   
'$
v_velџџџџџџџџџ   
'$
w_velџџџџџџџџџ   
p 

 
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
A__inference_model_layer_call_and_return_conditional_losses_134013ОnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЂ
Ђ
~{
'$
u_velџџџџџџџџџ   
'$
v_velџџџџџџџџџ   
'$
w_velџџџџџџџџџ   
p

 
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
A__inference_model_layer_call_and_return_conditional_losses_134970ЩnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЃЂ
Ђ

*'
inputs/0џџџџџџџџџ   
*'
inputs/1џџџџџџџџџ   
*'
inputs/2џџџџџџџџџ   
p 

 
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
A__inference_model_layer_call_and_return_conditional_losses_135526ЩnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЃЂ
Ђ

*'
inputs/0џџџџџџџџџ   
*'
inputs/1џџџџџџџџџ   
*'
inputs/2џџџџџџџџџ   
p

 
Њ "1Ђ.
'$
0џџџџџџџџџ   
 м
&__inference_model_layer_call_fn_132647БnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЂ
Ђ
~{
'$
u_velџџџџџџџџџ   
'$
v_velџџџџџџџџџ   
'$
w_velџџџџџџџџџ   
p 

 
Њ "$!џџџџџџџџџ   м
&__inference_model_layer_call_fn_133651БnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЂ
Ђ
~{
'$
u_velџџџџџџџџџ   
'$
v_velџџџџџџџџџ   
'$
w_velџџџџџџџџџ   
p

 
Њ "$!џџџџџџџџџ   ч
&__inference_model_layer_call_fn_134283МnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЃЂ
Ђ

*'
inputs/0џџџџџџџџџ   
*'
inputs/1џџџџџџџџџ   
*'
inputs/2џџџџџџџџџ   
p 

 
Њ "$!џџџџџџџџџ   ч
&__inference_model_layer_call_fn_134414МnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТЃЂ
Ђ

*'
inputs/0џџџџџџџџџ   
*'
inputs/1џџџџџџџџџ   
*'
inputs/2џџџџџџџџџ   
p

 
Њ "$!џџџџџџџџџ   Е
E__inference_reshape_1_layer_call_and_return_conditional_losses_135566l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_reshape_1_layer_call_fn_135551_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   Е
E__inference_reshape_2_layer_call_and_return_conditional_losses_135586l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
*__inference_reshape_2_layer_call_fn_135571_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   Ў
E__inference_reshape_6_layer_call_and_return_conditional_losses_136053e;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_reshape_6_layer_call_fn_136041X;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "џџџџџџџџџЎ
E__inference_reshape_7_layer_call_and_return_conditional_losses_136073e0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
*__inference_reshape_7_layer_call_fn_136058X0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџГ
C__inference_reshape_layer_call_and_return_conditional_losses_135546l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ   
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
(__inference_reshape_layer_call_fn_135531_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ   
Њ "$!џџџџџџџџџ   
$__inference_signature_wrapper_134152мnYZSTMNklef_`}~­ЎГДЙКзибвЫЬѕія№щъћќЅІ ЋЌБВЗИСТІЂЂ
Ђ 
Њ
0
u_vel'$
u_velџџџџџџџџџ   
0
v_vel'$
v_velџџџџџџџџџ   
0
w_vel'$
w_velџџџџџџџџџ   "AЊ>
<
	conv3d_30/,
	conv3d_30џџџџџџџџџ   П
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_136445p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
0__inference_up_sampling3d_1_layer_call_fn_136384c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ   П
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_136217p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
0__inference_up_sampling3d_2_layer_call_fn_136180c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_136511p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
0__inference_up_sampling3d_3_layer_call_fn_136450c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ   П
K__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_136259p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
0__inference_up_sampling3d_4_layer_call_fn_136222c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџП
K__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_136577p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ   
 
0__inference_up_sampling3d_5_layer_call_fn_136516c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ   Н
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_136175p;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "1Ђ.
'$
0џџџџџџџџџ
 
.__inference_up_sampling3d_layer_call_fn_136138c;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "$!џџџџџџџџџ