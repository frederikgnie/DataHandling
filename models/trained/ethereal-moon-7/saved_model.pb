��7
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
�
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
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��1
�
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
�
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
�
conv3d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_10/kernel
�
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
�
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
�
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
�
conv3d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_11/kernel
�
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
�
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
�
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
�
conv3d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_12/kernel
�
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
�
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
�
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
�
conv3d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_13/kernel
�
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
�
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
�
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
�
conv3d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_14/kernel
�
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
�
conv3d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_15/kernel
�
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
�
conv3d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_20/kernel
�
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
�
conv3d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_25/kernel
�
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
�
conv3d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_16/kernel
�
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
�
conv3d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_21/kernel
�
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
�
conv3d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_26/kernel
�
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
�
conv3d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_17/kernel
�
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
�
conv3d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_22/kernel
�
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
�
conv3d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_27/kernel
�
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
�
conv3d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_18/kernel
�
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
�
conv3d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_23/kernel
�
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
�
conv3d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_28/kernel
�
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
�
conv3d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_19/kernel
�
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
�
conv3d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_24/kernel
�
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
�
conv3d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_29/kernel
�
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
�
conv3d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_30/kernel
�
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
�
Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/m
�
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
�
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/m
�
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_10/kernel/m
�
+Adam/conv3d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_1/kernel/m
�
*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/m
�
*Adam/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_11/kernel/m
�
+Adam/conv3d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_2/kernel/m
�
*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/m
�
*Adam/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_12/kernel/m
�
+Adam/conv3d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/m
�
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_8/kernel/m
�
*Adam/conv3d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_13/kernel/m
�
+Adam/conv3d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_4/kernel/m
�
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_9/kernel/m
�
*Adam/conv3d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_14/kernel/m
�
+Adam/conv3d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_15/kernel/m
�
+Adam/conv3d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_20/kernel/m
�
+Adam/conv3d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_20/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_25/kernel/m
�
+Adam/conv3d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_25/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_16/kernel/m
�
+Adam/conv3d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_21/kernel/m
�
+Adam/conv3d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_21/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_26/kernel/m
�
+Adam/conv3d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_26/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_17/kernel/m
�
+Adam/conv3d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_22/kernel/m
�
+Adam/conv3d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_22/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_27/kernel/m
�
+Adam/conv3d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_27/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_18/kernel/m
�
+Adam/conv3d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_23/kernel/m
�
+Adam/conv3d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_23/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_28/kernel/m
�
+Adam/conv3d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_28/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_19/kernel/m
�
+Adam/conv3d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_24/kernel/m
�
+Adam/conv3d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_24/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_29/kernel/m
�
+Adam/conv3d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_29/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_30/kernel/m
�
+Adam/conv3d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_30/kernel/m**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/v
�
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
�
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/v
�
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_10/kernel/v
�
+Adam/conv3d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_1/kernel/v
�
*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/v
�
*Adam/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_11/kernel/v
�
+Adam/conv3d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_2/kernel/v
�
*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/v
�
*Adam/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_12/kernel/v
�
+Adam/conv3d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/v
�
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_8/kernel/v
�
*Adam/conv3d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_13/kernel/v
�
+Adam/conv3d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_4/kernel/v
�
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_9/kernel/v
�
*Adam/conv3d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_14/kernel/v
�
+Adam/conv3d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_15/kernel/v
�
+Adam/conv3d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_20/kernel/v
�
+Adam/conv3d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_20/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_25/kernel/v
�
+Adam/conv3d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_25/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_16/kernel/v
�
+Adam/conv3d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_21/kernel/v
�
+Adam/conv3d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_21/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_26/kernel/v
�
+Adam/conv3d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_26/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_17/kernel/v
�
+Adam/conv3d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_22/kernel/v
�
+Adam/conv3d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_22/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_27/kernel/v
�
+Adam/conv3d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_27/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_18/kernel/v
�
+Adam/conv3d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_23/kernel/v
�
+Adam/conv3d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_23/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_28/kernel/v
�
+Adam/conv3d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_28/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_19/kernel/v
�
+Adam/conv3d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_24/kernel/v
�
+Adam/conv3d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_24/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_29/kernel/v
�
+Adam/conv3d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_29/kernel/v**
_output_shapes
:*
dtype0
�
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
�
Adam/conv3d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_30/kernel/v
�
+Adam/conv3d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_30/kernel/v**
_output_shapes
:*
dtype0
�
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
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer-22
layer-23
layer-24
layer_with_weights-15
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-18
 layer-31
!layer_with_weights-19
!layer-32
"layer_with_weights-20
"layer-33
#layer_with_weights-21
#layer-34
$layer_with_weights-22
$layer-35
%layer_with_weights-23
%layer-36
&layer-37
'layer-38
(layer-39
)layer_with_weights-24
)layer-40
*layer_with_weights-25
*layer-41
+layer_with_weights-26
+layer-42
,layer_with_weights-27
,layer-43
-layer_with_weights-28
-layer-44
.layer_with_weights-29
.layer-45
/layer-46
0layer_with_weights-30
0layer-47
1	optimizer
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6
signatures
 
h

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
R
[	variables
\trainable_variables
]regularization_losses
^	keras_api
R
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
h

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
m

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�

	�iter
�beta_1
�beta_2

�decay
�learning_rate7m�8m�=m�>m�Cm�Dm�Im�Jm�Om�Pm�Um�Vm�gm�hm�mm�nm�sm�tm�ym�zm�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�7v�8v�=v�>v�Cv�Dv�Iv�Jv�Ov�Pv�Uv�Vv�gv�hv�mv�nv�sv�tv�yv�zv�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
�
70
81
=2
>3
C4
D5
I6
J7
O8
P9
U10
V11
g12
h13
m14
n15
s16
t17
y18
z19
20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�
70
81
=2
>3
C4
D5
I6
J7
O8
P9
U10
V11
g12
h13
m14
n15
s16
t17
y18
z19
20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
\Z
VARIABLE_VALUEconv3d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
[Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
\Z
VARIABLE_VALUEconv3d_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
[Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
\Z
VARIABLE_VALUEconv3d_12/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_12/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
\Z
VARIABLE_VALUEconv3d_8/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_8/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
�1

0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_13/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_13/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
\Z
VARIABLE_VALUEconv3d_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_4/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
\Z
VARIABLE_VALUEconv3d_9/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_9/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_14/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_14/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_15/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_15/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_20/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_20/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_25/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_25/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_16/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_16/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_21/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_21/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_26/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_26/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_17/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_17/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_22/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_22/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_27/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_27/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_18/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_18/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_23/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_23/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_28/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_28/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_19/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_19/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_24/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_24/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_29/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_29/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_30/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_30/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
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
�
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

�0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

�total

�count
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
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
�~
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
�~
VARIABLE_VALUEAdam/conv3d_14/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_14/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_15/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_15/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_20/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_20/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_25/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_25/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_16/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_16/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_21/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_21/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_26/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_26/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_17/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_17/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_22/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_22/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_27/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_27/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_18/kernel/mSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_18/bias/mQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_23/kernel/mSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_23/bias/mQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_28/kernel/mSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_28/bias/mQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_19/kernel/mSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_19/bias/mQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_24/kernel/mSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_24/bias/mQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_29/kernel/mSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_29/bias/mQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
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
�~
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
�~
VARIABLE_VALUEAdam/conv3d_14/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_14/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_15/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_15/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_20/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_20/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_25/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_25/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_16/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_16/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_21/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_21/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_26/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_26/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_17/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_17/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_22/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_22/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_27/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_27/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_18/kernel/vSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_18/bias/vQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_23/kernel/vSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_23/bias/vQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_28/kernel/vSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_28/bias/vQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_19/kernel/vSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_19/bias/vQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_24/kernel/vSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_24/bias/vQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_29/kernel/vSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_29/bias/vQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_30/kernel/vSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_30/bias/vQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*3
_output_shapes!
:���������   *
dtype0*(
shape:���������   
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d_10/kernelconv3d_10/biasconv3d_5/kernelconv3d_5/biasconv3d/kernelconv3d/biasconv3d_11/kernelconv3d_11/biasconv3d_6/kernelconv3d_6/biasconv3d_1/kernelconv3d_1/biasconv3d_12/kernelconv3d_12/biasconv3d_7/kernelconv3d_7/biasconv3d_2/kernelconv3d_2/biasconv3d_13/kernelconv3d_13/biasconv3d_8/kernelconv3d_8/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_9/kernelconv3d_9/biasconv3d_14/kernelconv3d_14/biasconv3d_25/kernelconv3d_25/biasconv3d_20/kernelconv3d_20/biasconv3d_15/kernelconv3d_15/biasconv3d_26/kernelconv3d_26/biasconv3d_21/kernelconv3d_21/biasconv3d_16/kernelconv3d_16/biasconv3d_27/kernelconv3d_27/biasconv3d_22/kernelconv3d_22/biasconv3d_17/kernelconv3d_17/biasconv3d_28/kernelconv3d_28/biasconv3d_23/kernelconv3d_23/biasconv3d_18/kernelconv3d_18/biasconv3d_19/kernelconv3d_19/biasconv3d_24/kernelconv3d_24/biasconv3d_29/kernelconv3d_29/biasconv3d_30/kernelconv3d_30/bias*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_93808
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�A
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp$conv3d_10/kernel/Read/ReadVariableOp"conv3d_10/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp$conv3d_11/kernel/Read/ReadVariableOp"conv3d_11/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp$conv3d_12/kernel/Read/ReadVariableOp"conv3d_12/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_8/kernel/Read/ReadVariableOp!conv3d_8/bias/Read/ReadVariableOp$conv3d_13/kernel/Read/ReadVariableOp"conv3d_13/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_9/kernel/Read/ReadVariableOp!conv3d_9/bias/Read/ReadVariableOp$conv3d_14/kernel/Read/ReadVariableOp"conv3d_14/bias/Read/ReadVariableOp$conv3d_15/kernel/Read/ReadVariableOp"conv3d_15/bias/Read/ReadVariableOp$conv3d_20/kernel/Read/ReadVariableOp"conv3d_20/bias/Read/ReadVariableOp$conv3d_25/kernel/Read/ReadVariableOp"conv3d_25/bias/Read/ReadVariableOp$conv3d_16/kernel/Read/ReadVariableOp"conv3d_16/bias/Read/ReadVariableOp$conv3d_21/kernel/Read/ReadVariableOp"conv3d_21/bias/Read/ReadVariableOp$conv3d_26/kernel/Read/ReadVariableOp"conv3d_26/bias/Read/ReadVariableOp$conv3d_17/kernel/Read/ReadVariableOp"conv3d_17/bias/Read/ReadVariableOp$conv3d_22/kernel/Read/ReadVariableOp"conv3d_22/bias/Read/ReadVariableOp$conv3d_27/kernel/Read/ReadVariableOp"conv3d_27/bias/Read/ReadVariableOp$conv3d_18/kernel/Read/ReadVariableOp"conv3d_18/bias/Read/ReadVariableOp$conv3d_23/kernel/Read/ReadVariableOp"conv3d_23/bias/Read/ReadVariableOp$conv3d_28/kernel/Read/ReadVariableOp"conv3d_28/bias/Read/ReadVariableOp$conv3d_19/kernel/Read/ReadVariableOp"conv3d_19/bias/Read/ReadVariableOp$conv3d_24/kernel/Read/ReadVariableOp"conv3d_24/bias/Read/ReadVariableOp$conv3d_29/kernel/Read/ReadVariableOp"conv3d_29/bias/Read/ReadVariableOp$conv3d_30/kernel/Read/ReadVariableOp"conv3d_30/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp+Adam/conv3d_10/kernel/m/Read/ReadVariableOp)Adam/conv3d_10/bias/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp*Adam/conv3d_6/kernel/m/Read/ReadVariableOp(Adam/conv3d_6/bias/m/Read/ReadVariableOp+Adam/conv3d_11/kernel/m/Read/ReadVariableOp)Adam/conv3d_11/bias/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp*Adam/conv3d_7/kernel/m/Read/ReadVariableOp(Adam/conv3d_7/bias/m/Read/ReadVariableOp+Adam/conv3d_12/kernel/m/Read/ReadVariableOp)Adam/conv3d_12/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_8/kernel/m/Read/ReadVariableOp(Adam/conv3d_8/bias/m/Read/ReadVariableOp+Adam/conv3d_13/kernel/m/Read/ReadVariableOp)Adam/conv3d_13/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/conv3d_9/kernel/m/Read/ReadVariableOp(Adam/conv3d_9/bias/m/Read/ReadVariableOp+Adam/conv3d_14/kernel/m/Read/ReadVariableOp)Adam/conv3d_14/bias/m/Read/ReadVariableOp+Adam/conv3d_15/kernel/m/Read/ReadVariableOp)Adam/conv3d_15/bias/m/Read/ReadVariableOp+Adam/conv3d_20/kernel/m/Read/ReadVariableOp)Adam/conv3d_20/bias/m/Read/ReadVariableOp+Adam/conv3d_25/kernel/m/Read/ReadVariableOp)Adam/conv3d_25/bias/m/Read/ReadVariableOp+Adam/conv3d_16/kernel/m/Read/ReadVariableOp)Adam/conv3d_16/bias/m/Read/ReadVariableOp+Adam/conv3d_21/kernel/m/Read/ReadVariableOp)Adam/conv3d_21/bias/m/Read/ReadVariableOp+Adam/conv3d_26/kernel/m/Read/ReadVariableOp)Adam/conv3d_26/bias/m/Read/ReadVariableOp+Adam/conv3d_17/kernel/m/Read/ReadVariableOp)Adam/conv3d_17/bias/m/Read/ReadVariableOp+Adam/conv3d_22/kernel/m/Read/ReadVariableOp)Adam/conv3d_22/bias/m/Read/ReadVariableOp+Adam/conv3d_27/kernel/m/Read/ReadVariableOp)Adam/conv3d_27/bias/m/Read/ReadVariableOp+Adam/conv3d_18/kernel/m/Read/ReadVariableOp)Adam/conv3d_18/bias/m/Read/ReadVariableOp+Adam/conv3d_23/kernel/m/Read/ReadVariableOp)Adam/conv3d_23/bias/m/Read/ReadVariableOp+Adam/conv3d_28/kernel/m/Read/ReadVariableOp)Adam/conv3d_28/bias/m/Read/ReadVariableOp+Adam/conv3d_19/kernel/m/Read/ReadVariableOp)Adam/conv3d_19/bias/m/Read/ReadVariableOp+Adam/conv3d_24/kernel/m/Read/ReadVariableOp)Adam/conv3d_24/bias/m/Read/ReadVariableOp+Adam/conv3d_29/kernel/m/Read/ReadVariableOp)Adam/conv3d_29/bias/m/Read/ReadVariableOp+Adam/conv3d_30/kernel/m/Read/ReadVariableOp)Adam/conv3d_30/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp+Adam/conv3d_10/kernel/v/Read/ReadVariableOp)Adam/conv3d_10/bias/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp*Adam/conv3d_6/kernel/v/Read/ReadVariableOp(Adam/conv3d_6/bias/v/Read/ReadVariableOp+Adam/conv3d_11/kernel/v/Read/ReadVariableOp)Adam/conv3d_11/bias/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp*Adam/conv3d_7/kernel/v/Read/ReadVariableOp(Adam/conv3d_7/bias/v/Read/ReadVariableOp+Adam/conv3d_12/kernel/v/Read/ReadVariableOp)Adam/conv3d_12/bias/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_8/kernel/v/Read/ReadVariableOp(Adam/conv3d_8/bias/v/Read/ReadVariableOp+Adam/conv3d_13/kernel/v/Read/ReadVariableOp)Adam/conv3d_13/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/conv3d_9/kernel/v/Read/ReadVariableOp(Adam/conv3d_9/bias/v/Read/ReadVariableOp+Adam/conv3d_14/kernel/v/Read/ReadVariableOp)Adam/conv3d_14/bias/v/Read/ReadVariableOp+Adam/conv3d_15/kernel/v/Read/ReadVariableOp)Adam/conv3d_15/bias/v/Read/ReadVariableOp+Adam/conv3d_20/kernel/v/Read/ReadVariableOp)Adam/conv3d_20/bias/v/Read/ReadVariableOp+Adam/conv3d_25/kernel/v/Read/ReadVariableOp)Adam/conv3d_25/bias/v/Read/ReadVariableOp+Adam/conv3d_16/kernel/v/Read/ReadVariableOp)Adam/conv3d_16/bias/v/Read/ReadVariableOp+Adam/conv3d_21/kernel/v/Read/ReadVariableOp)Adam/conv3d_21/bias/v/Read/ReadVariableOp+Adam/conv3d_26/kernel/v/Read/ReadVariableOp)Adam/conv3d_26/bias/v/Read/ReadVariableOp+Adam/conv3d_17/kernel/v/Read/ReadVariableOp)Adam/conv3d_17/bias/v/Read/ReadVariableOp+Adam/conv3d_22/kernel/v/Read/ReadVariableOp)Adam/conv3d_22/bias/v/Read/ReadVariableOp+Adam/conv3d_27/kernel/v/Read/ReadVariableOp)Adam/conv3d_27/bias/v/Read/ReadVariableOp+Adam/conv3d_18/kernel/v/Read/ReadVariableOp)Adam/conv3d_18/bias/v/Read/ReadVariableOp+Adam/conv3d_23/kernel/v/Read/ReadVariableOp)Adam/conv3d_23/bias/v/Read/ReadVariableOp+Adam/conv3d_28/kernel/v/Read/ReadVariableOp)Adam/conv3d_28/bias/v/Read/ReadVariableOp+Adam/conv3d_19/kernel/v/Read/ReadVariableOp)Adam/conv3d_19/bias/v/Read/ReadVariableOp+Adam/conv3d_24/kernel/v/Read/ReadVariableOp)Adam/conv3d_24/bias/v/Read/ReadVariableOp+Adam/conv3d_29/kernel/v/Read/ReadVariableOp)Adam/conv3d_29/bias/v/Read/ReadVariableOp+Adam/conv3d_30/kernel/v/Read/ReadVariableOp)Adam/conv3d_30/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_96836
�#
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_5/kernelconv3d_5/biasconv3d_10/kernelconv3d_10/biasconv3d_1/kernelconv3d_1/biasconv3d_6/kernelconv3d_6/biasconv3d_11/kernelconv3d_11/biasconv3d_2/kernelconv3d_2/biasconv3d_7/kernelconv3d_7/biasconv3d_12/kernelconv3d_12/biasconv3d_3/kernelconv3d_3/biasconv3d_8/kernelconv3d_8/biasconv3d_13/kernelconv3d_13/biasconv3d_4/kernelconv3d_4/biasconv3d_9/kernelconv3d_9/biasconv3d_14/kernelconv3d_14/biasconv3d_15/kernelconv3d_15/biasconv3d_20/kernelconv3d_20/biasconv3d_25/kernelconv3d_25/biasconv3d_16/kernelconv3d_16/biasconv3d_21/kernelconv3d_21/biasconv3d_26/kernelconv3d_26/biasconv3d_17/kernelconv3d_17/biasconv3d_22/kernelconv3d_22/biasconv3d_27/kernelconv3d_27/biasconv3d_18/kernelconv3d_18/biasconv3d_23/kernelconv3d_23/biasconv3d_28/kernelconv3d_28/biasconv3d_19/kernelconv3d_19/biasconv3d_24/kernelconv3d_24/biasconv3d_29/kernelconv3d_29/biasconv3d_30/kernelconv3d_30/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/mAdam/conv3d_10/kernel/mAdam/conv3d_10/bias/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/mAdam/conv3d_6/kernel/mAdam/conv3d_6/bias/mAdam/conv3d_11/kernel/mAdam/conv3d_11/bias/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/mAdam/conv3d_7/kernel/mAdam/conv3d_7/bias/mAdam/conv3d_12/kernel/mAdam/conv3d_12/bias/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_8/kernel/mAdam/conv3d_8/bias/mAdam/conv3d_13/kernel/mAdam/conv3d_13/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/conv3d_9/kernel/mAdam/conv3d_9/bias/mAdam/conv3d_14/kernel/mAdam/conv3d_14/bias/mAdam/conv3d_15/kernel/mAdam/conv3d_15/bias/mAdam/conv3d_20/kernel/mAdam/conv3d_20/bias/mAdam/conv3d_25/kernel/mAdam/conv3d_25/bias/mAdam/conv3d_16/kernel/mAdam/conv3d_16/bias/mAdam/conv3d_21/kernel/mAdam/conv3d_21/bias/mAdam/conv3d_26/kernel/mAdam/conv3d_26/bias/mAdam/conv3d_17/kernel/mAdam/conv3d_17/bias/mAdam/conv3d_22/kernel/mAdam/conv3d_22/bias/mAdam/conv3d_27/kernel/mAdam/conv3d_27/bias/mAdam/conv3d_18/kernel/mAdam/conv3d_18/bias/mAdam/conv3d_23/kernel/mAdam/conv3d_23/bias/mAdam/conv3d_28/kernel/mAdam/conv3d_28/bias/mAdam/conv3d_19/kernel/mAdam/conv3d_19/bias/mAdam/conv3d_24/kernel/mAdam/conv3d_24/bias/mAdam/conv3d_29/kernel/mAdam/conv3d_29/bias/mAdam/conv3d_30/kernel/mAdam/conv3d_30/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/vAdam/conv3d_10/kernel/vAdam/conv3d_10/bias/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/vAdam/conv3d_6/kernel/vAdam/conv3d_6/bias/vAdam/conv3d_11/kernel/vAdam/conv3d_11/bias/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/vAdam/conv3d_7/kernel/vAdam/conv3d_7/bias/vAdam/conv3d_12/kernel/vAdam/conv3d_12/bias/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_8/kernel/vAdam/conv3d_8/bias/vAdam/conv3d_13/kernel/vAdam/conv3d_13/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/conv3d_9/kernel/vAdam/conv3d_9/bias/vAdam/conv3d_14/kernel/vAdam/conv3d_14/bias/vAdam/conv3d_15/kernel/vAdam/conv3d_15/bias/vAdam/conv3d_20/kernel/vAdam/conv3d_20/bias/vAdam/conv3d_25/kernel/vAdam/conv3d_25/bias/vAdam/conv3d_16/kernel/vAdam/conv3d_16/bias/vAdam/conv3d_21/kernel/vAdam/conv3d_21/bias/vAdam/conv3d_26/kernel/vAdam/conv3d_26/bias/vAdam/conv3d_17/kernel/vAdam/conv3d_17/bias/vAdam/conv3d_22/kernel/vAdam/conv3d_22/bias/vAdam/conv3d_27/kernel/vAdam/conv3d_27/bias/vAdam/conv3d_18/kernel/vAdam/conv3d_18/bias/vAdam/conv3d_23/kernel/vAdam/conv3d_23/bias/vAdam/conv3d_28/kernel/vAdam/conv3d_28/bias/vAdam/conv3d_19/kernel/vAdam/conv3d_19/bias/vAdam/conv3d_24/kernel/vAdam/conv3d_24/bias/vAdam/conv3d_29/kernel/vAdam/conv3d_29/bias/vAdam/conv3d_30/kernel/vAdam/conv3d_30/bias/v*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_97425��+
�
f
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_95284

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
z
@__inference_add_1_layer_call_and_return_conditional_losses_96215
inputs_0
inputs_1
inputs_2
identity^
addAddV2inputs_0inputs_1*
T0*3
_output_shapes!
:���������   _
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:���������   ]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:���������   :���������   :���������   :] Y
3
_output_shapes!
:���������   
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������   
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:���������   
"
_user_specified_name
inputs/2
�
�
%__inference_model_layer_call_fn_94066

inputs%
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
identity��StatefulPartitionedCall�	
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
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_93065{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_95576

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
valueB:�
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
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_19_layer_call_fn_96149

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_19_layer_call_and_return_conditional_losses_92161{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_26_layer_call_and_return_conditional_losses_91819

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_11_layer_call_fn_95213

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_91370{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_91559

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:���������c
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_11_layer_call_and_return_conditional_losses_91370

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_95304

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
I
-__inference_max_pooling3d_layer_call_fn_95229

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91238�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91546

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_93671
input_1-
conv3d_10_93499:
conv3d_10_93501:,
conv3d_5_93504:
conv3d_5_93506:*
conv3d_93509:
conv3d_93511:-
conv3d_11_93514:
conv3d_11_93516:,
conv3d_6_93519:
conv3d_6_93521:,
conv3d_1_93524:
conv3d_1_93526:-
conv3d_12_93532:
conv3d_12_93534:,
conv3d_7_93537:
conv3d_7_93539:,
conv3d_2_93542:
conv3d_2_93544:-
conv3d_13_93547:
conv3d_13_93549:,
conv3d_8_93552:
conv3d_8_93554:,
conv3d_3_93557:
conv3d_3_93559:,
conv3d_4_93565:
conv3d_4_93567:,
conv3d_9_93570:
conv3d_9_93572:-
conv3d_14_93575:
conv3d_14_93577:-
conv3d_25_93583:
conv3d_25_93585:-
conv3d_20_93588:
conv3d_20_93590:-
conv3d_15_93593:
conv3d_15_93595:-
conv3d_26_93601:
conv3d_26_93603:-
conv3d_21_93606:
conv3d_21_93608:-
conv3d_16_93611:
conv3d_16_93613:-
conv3d_27_93616:
conv3d_27_93618:-
conv3d_22_93621:
conv3d_22_93623:-
conv3d_17_93626:
conv3d_17_93628:-
conv3d_28_93634:
conv3d_28_93636:-
conv3d_23_93639:
conv3d_23_93641:-
conv3d_18_93644:
conv3d_18_93646:-
conv3d_19_93649:
conv3d_19_93651:-
conv3d_24_93654:
conv3d_24_93656:-
conv3d_29_93659:
conv3d_29_93661:-
conv3d_30_93665:
conv3d_30_93667:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall�!conv3d_18/StatefulPartitionedCall�!conv3d_19/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall�!conv3d_20/StatefulPartitionedCall�!conv3d_21/StatefulPartitionedCall�!conv3d_22/StatefulPartitionedCall�!conv3d_23/StatefulPartitionedCall�!conv3d_24/StatefulPartitionedCall�!conv3d_25/StatefulPartitionedCall�!conv3d_26/StatefulPartitionedCall�!conv3d_27/StatefulPartitionedCall�!conv3d_28/StatefulPartitionedCall�!conv3d_29/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�!conv3d_30/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_10_93499conv3d_10_93501*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_91319�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_5_93504conv3d_5_93506*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_91336�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_93509conv3d_93511*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_91353�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_93514conv3d_11_93516*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_91370�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_93519conv3d_6_93521*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_91387�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_93524conv3d_1_93526*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_91404�
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91414�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91420�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91426�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_93532conv3d_12_93534*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_91439�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_93537conv3d_7_93539*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_91456�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_93542conv3d_2_93544*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_91473�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_93547conv3d_13_93549*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_91490�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_93552conv3d_8_93554*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_91507�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_93557conv3d_3_93559*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_91524�
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91534�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91540�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91546�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_93565conv3d_4_93567*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_91559�
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_93570conv3d_9_93572*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_91576�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_93575conv3d_14_93577*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_91593�
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_91607�
reshape/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_91621�
reshape_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_91638�
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_25_93583conv3d_25_93585*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_25_layer_call_and_return_conditional_losses_91651�
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_20_93588conv3d_20_93590*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_20_layer_call_and_return_conditional_losses_91668�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_15_93593conv3d_15_93595*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_91685�
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_91728�
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_91767�
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_91806�
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_93601conv3d_26_93603*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_26_layer_call_and_return_conditional_losses_91819�
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_93606conv3d_21_93608*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_21_layer_call_and_return_conditional_losses_91836�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_93611conv3d_16_93613*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_91853�
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_93616conv3d_27_93618*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_27_layer_call_and_return_conditional_losses_91870�
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_93621conv3d_22_93623*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_22_layer_call_and_return_conditional_losses_91887�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_93626conv3d_17_93628*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_91904�
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_91971�
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_92034�
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_92097�
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_93634conv3d_28_93636*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_28_layer_call_and_return_conditional_losses_92110�
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_93639conv3d_23_93641*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_23_layer_call_and_return_conditional_losses_92127�
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_93644conv3d_18_93646*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_92144�
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_93649conv3d_19_93651*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_19_layer_call_and_return_conditional_losses_92161�
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_93654conv3d_24_93656*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_24_layer_call_and_return_conditional_losses_92178�
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_93659conv3d_29_93661*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_29_layer_call_and_return_conditional_losses_92195�
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_92209�
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_93665conv3d_30_93667*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_30_layer_call_and_return_conditional_losses_92221�
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   �	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
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
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
)__inference_conv3d_24_layer_call_fn_96169

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_24_layer_call_and_return_conditional_losses_92178{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_17_layer_call_and_return_conditional_losses_91904

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
A__inference_conv3d_layer_call_and_return_conditional_losses_95124

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_10_layer_call_fn_95153

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_91319{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_5_layer_call_fn_95454

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91534l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_17_layer_call_fn_95831

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_91904{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_26_layer_call_and_return_conditional_losses_95822

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_6_layer_call_fn_95193

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_91387{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91274

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_95464

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_8_layer_call_and_return_conditional_losses_91507

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91238

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_21_layer_call_fn_95791

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_21_layer_call_and_return_conditional_losses_91836{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_28_layer_call_and_return_conditional_losses_92110

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_95364

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
&__inference_conv3d_layer_call_fn_95113

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_91353{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_95264

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_23_layer_call_and_return_conditional_losses_92127

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�%
f
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_95948

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_26_layer_call_fn_95811

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_26_layer_call_and_return_conditional_losses_91819{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�%
f
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_91971

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
_
%__inference_add_1_layer_call_fn_96207
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_92209l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:���������   :���������   :���������   :] Y
3
_output_shapes!
:���������   
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������   
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:���������   
"
_user_specified_name
inputs/2
�%
f
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_92097

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_5_layer_call_fn_95449

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91298�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_13_layer_call_and_return_conditional_losses_91490

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�	
^
B__inference_reshape_layer_call_and_return_conditional_losses_95556

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
valueB:�
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
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_95720

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_95279

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91540

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_95144

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_14_layer_call_and_return_conditional_losses_91593

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:���������c
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_up_sampling3d_1_layer_call_fn_95887

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_92097l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_10_layer_call_and_return_conditional_losses_95164

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_18_layer_call_and_return_conditional_losses_96100

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_21_layer_call_and_return_conditional_losses_95802

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_14_layer_call_and_return_conditional_losses_95524

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:���������c
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_23_layer_call_and_return_conditional_losses_96120

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_30_layer_call_fn_96224

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_30_layer_call_and_return_conditional_losses_92221{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91426

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_93065

inputs-
conv3d_10_92893:
conv3d_10_92895:,
conv3d_5_92898:
conv3d_5_92900:*
conv3d_92903:
conv3d_92905:-
conv3d_11_92908:
conv3d_11_92910:,
conv3d_6_92913:
conv3d_6_92915:,
conv3d_1_92918:
conv3d_1_92920:-
conv3d_12_92926:
conv3d_12_92928:,
conv3d_7_92931:
conv3d_7_92933:,
conv3d_2_92936:
conv3d_2_92938:-
conv3d_13_92941:
conv3d_13_92943:,
conv3d_8_92946:
conv3d_8_92948:,
conv3d_3_92951:
conv3d_3_92953:,
conv3d_4_92959:
conv3d_4_92961:,
conv3d_9_92964:
conv3d_9_92966:-
conv3d_14_92969:
conv3d_14_92971:-
conv3d_25_92977:
conv3d_25_92979:-
conv3d_20_92982:
conv3d_20_92984:-
conv3d_15_92987:
conv3d_15_92989:-
conv3d_26_92995:
conv3d_26_92997:-
conv3d_21_93000:
conv3d_21_93002:-
conv3d_16_93005:
conv3d_16_93007:-
conv3d_27_93010:
conv3d_27_93012:-
conv3d_22_93015:
conv3d_22_93017:-
conv3d_17_93020:
conv3d_17_93022:-
conv3d_28_93028:
conv3d_28_93030:-
conv3d_23_93033:
conv3d_23_93035:-
conv3d_18_93038:
conv3d_18_93040:-
conv3d_19_93043:
conv3d_19_93045:-
conv3d_24_93048:
conv3d_24_93050:-
conv3d_29_93053:
conv3d_29_93055:-
conv3d_30_93059:
conv3d_30_93061:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall�!conv3d_18/StatefulPartitionedCall�!conv3d_19/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall�!conv3d_20/StatefulPartitionedCall�!conv3d_21/StatefulPartitionedCall�!conv3d_22/StatefulPartitionedCall�!conv3d_23/StatefulPartitionedCall�!conv3d_24/StatefulPartitionedCall�!conv3d_25/StatefulPartitionedCall�!conv3d_26/StatefulPartitionedCall�!conv3d_27/StatefulPartitionedCall�!conv3d_28/StatefulPartitionedCall�!conv3d_29/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�!conv3d_30/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_10_92893conv3d_10_92895*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_91319�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_5_92898conv3d_5_92900*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_91336�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_92903conv3d_92905*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_91353�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_92908conv3d_11_92910*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_91370�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_92913conv3d_6_92915*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_91387�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_92918conv3d_1_92920*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_91404�
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91414�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91420�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91426�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_92926conv3d_12_92928*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_91439�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_92931conv3d_7_92933*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_91456�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_92936conv3d_2_92938*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_91473�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_92941conv3d_13_92943*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_91490�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_92946conv3d_8_92948*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_91507�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_92951conv3d_3_92953*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_91524�
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91534�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91540�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91546�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_92959conv3d_4_92961*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_91559�
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_92964conv3d_9_92966*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_91576�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_92969conv3d_14_92971*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_91593�
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_91607�
reshape/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_91621�
reshape_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_91638�
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_25_92977conv3d_25_92979*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_25_layer_call_and_return_conditional_losses_91651�
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_20_92982conv3d_20_92984*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_20_layer_call_and_return_conditional_losses_91668�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_15_92987conv3d_15_92989*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_91685�
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_91728�
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_91767�
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_91806�
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_92995conv3d_26_92997*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_26_layer_call_and_return_conditional_losses_91819�
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_93000conv3d_21_93002*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_21_layer_call_and_return_conditional_losses_91836�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_93005conv3d_16_93007*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_91853�
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_93010conv3d_27_93012*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_27_layer_call_and_return_conditional_losses_91870�
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_93015conv3d_22_93017*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_22_layer_call_and_return_conditional_losses_91887�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_93020conv3d_17_93022*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_91904�
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_91971�
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_92034�
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_92097�
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_93028conv3d_28_93030*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_28_layer_call_and_return_conditional_losses_92110�
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_93033conv3d_23_93035*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_23_layer_call_and_return_conditional_losses_92127�
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_93038conv3d_18_93040*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_92144�
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_93043conv3d_19_93045*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_19_layer_call_and_return_conditional_losses_92161�
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_93048conv3d_24_93050*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_24_layer_call_and_return_conditional_losses_92178�
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_93053conv3d_29_93055*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_29_layer_call_and_return_conditional_losses_92195�
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_92209�
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_93059conv3d_30_93061*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_30_layer_call_and_return_conditional_losses_92221�
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   �	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
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
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_2_layer_call_fn_95254

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91420l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_9_layer_call_and_return_conditional_losses_95504

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:���������c
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
ߩ
�0
@__inference_model_layer_call_and_return_conditional_losses_94585

inputsF
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
identity��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp� conv3d_10/BiasAdd/ReadVariableOp�conv3d_10/Conv3D/ReadVariableOp� conv3d_11/BiasAdd/ReadVariableOp�conv3d_11/Conv3D/ReadVariableOp� conv3d_12/BiasAdd/ReadVariableOp�conv3d_12/Conv3D/ReadVariableOp� conv3d_13/BiasAdd/ReadVariableOp�conv3d_13/Conv3D/ReadVariableOp� conv3d_14/BiasAdd/ReadVariableOp�conv3d_14/Conv3D/ReadVariableOp� conv3d_15/BiasAdd/ReadVariableOp�conv3d_15/Conv3D/ReadVariableOp� conv3d_16/BiasAdd/ReadVariableOp�conv3d_16/Conv3D/ReadVariableOp� conv3d_17/BiasAdd/ReadVariableOp�conv3d_17/Conv3D/ReadVariableOp� conv3d_18/BiasAdd/ReadVariableOp�conv3d_18/Conv3D/ReadVariableOp� conv3d_19/BiasAdd/ReadVariableOp�conv3d_19/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp� conv3d_20/BiasAdd/ReadVariableOp�conv3d_20/Conv3D/ReadVariableOp� conv3d_21/BiasAdd/ReadVariableOp�conv3d_21/Conv3D/ReadVariableOp� conv3d_22/BiasAdd/ReadVariableOp�conv3d_22/Conv3D/ReadVariableOp� conv3d_23/BiasAdd/ReadVariableOp�conv3d_23/Conv3D/ReadVariableOp� conv3d_24/BiasAdd/ReadVariableOp�conv3d_24/Conv3D/ReadVariableOp� conv3d_25/BiasAdd/ReadVariableOp�conv3d_25/Conv3D/ReadVariableOp� conv3d_26/BiasAdd/ReadVariableOp�conv3d_26/Conv3D/ReadVariableOp� conv3d_27/BiasAdd/ReadVariableOp�conv3d_27/Conv3D/ReadVariableOp� conv3d_28/BiasAdd/ReadVariableOp�conv3d_28/Conv3D/ReadVariableOp� conv3d_29/BiasAdd/ReadVariableOp�conv3d_29/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp� conv3d_30/BiasAdd/ReadVariableOp�conv3d_30/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�conv3d_8/BiasAdd/ReadVariableOp�conv3d_8/Conv3D/ReadVariableOp�conv3d_9/BiasAdd/ReadVariableOp�conv3d_9/Conv3D/ReadVariableOp�
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_10/Conv3DConv3Dinputs'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_10/ReluReluconv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_5/Conv3DConv3Dinputs&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_11/Conv3DConv3Dconv3d_10/Relu:activations:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_11/ReluReluconv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_6/Conv3DConv3Dconv3d_5/Relu:activations:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
max_pooling3d_4/MaxPool3D	MaxPool3Dconv3d_11/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_12/Conv3DConv3D"max_pooling3d_4/MaxPool3D:output:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_12/ReluReluconv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_7/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_13/Conv3DConv3Dconv3d_12/Relu:activations:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_13/ReluReluconv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_8/Conv3DConv3Dconv3d_7/Relu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_8/ReluReluconv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
max_pooling3d_5/MaxPool3D	MaxPool3Dconv3d_13/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_4/TanhTanhconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_9/Conv3DConv3D"max_pooling3d_3/MaxPool3D:output:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_9/TanhTanhconv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_14/Conv3DConv3D"max_pooling3d_5/MaxPool3D:output:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_14/TanhTanhconv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:���������t
add/addAddV2conv3d_4/Tanh:y:0conv3d_9/Tanh:y:0*
T0*3
_output_shapes!
:���������q
	add/add_1AddV2add/add:z:0conv3d_14/Tanh:y:0*
T0*3
_output_shapes!
:���������J
reshape/ShapeShapeadd/add_1:z:0*
T0*
_output_shapes
:e
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
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:|
reshape/ReshapeReshapeadd/add_1:z:0reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:����������W
reshape_1/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:g
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
valueB:�
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
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapereshape/Reshape:output:0 reshape_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:����������
conv3d_25/Conv3D/ReadVariableOpReadVariableOp(conv3d_25_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_25/Conv3DConv3Dreshape_1/Reshape:output:0'conv3d_25/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_25/BiasAdd/ReadVariableOpReadVariableOp)conv3d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_25/BiasAddBiasAddconv3d_25/Conv3D:output:0(conv3d_25/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_25/ReluReluconv3d_25/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_20/Conv3D/ReadVariableOpReadVariableOp(conv3d_20_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_20/Conv3DConv3Dreshape_1/Reshape:output:0'conv3d_20/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_20/BiasAdd/ReadVariableOpReadVariableOp)conv3d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_20/BiasAddBiasAddconv3d_20/Conv3D:output:0(conv3d_20/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_20/ReluReluconv3d_20/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_15/Conv3DConv3Dreshape_1/Reshape:output:0'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:���������a
up_sampling3d_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/splitSplit(up_sampling3d_4/split/split_dim:output:0conv3d_25/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/concatConcatV2up_sampling3d_4/split:output:0up_sampling3d_4/split:output:0up_sampling3d_4/split:output:1up_sampling3d_4/split:output:1up_sampling3d_4/split:output:2up_sampling3d_4/split:output:2up_sampling3d_4/split:output:3up_sampling3d_4/split:output:3up_sampling3d_4/split:output:4up_sampling3d_4/split:output:4up_sampling3d_4/split:output:5up_sampling3d_4/split:output:5up_sampling3d_4/split:output:6up_sampling3d_4/split:output:6up_sampling3d_4/split:output:7up_sampling3d_4/split:output:7$up_sampling3d_4/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/split_1Split*up_sampling3d_4/split_1/split_dim:output:0up_sampling3d_4/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/concat_1ConcatV2 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:7 up_sampling3d_4/split_1:output:7&up_sampling3d_4/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_4/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/split_2Split*up_sampling3d_4/split_2/split_dim:output:0!up_sampling3d_4/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/concat_2ConcatV2 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:7 up_sampling3d_4/split_2:output:7&up_sampling3d_4/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������a
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0conv3d_20/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7$up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7&up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7&up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������_
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0conv3d_15/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split[
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7"up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������a
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7$up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������a
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7$up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:����������
conv3d_26/Conv3D/ReadVariableOpReadVariableOp(conv3d_26_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_26/Conv3DConv3D!up_sampling3d_4/concat_2:output:0'conv3d_26/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_26/BiasAdd/ReadVariableOpReadVariableOp)conv3d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_26/BiasAddBiasAddconv3d_26/Conv3D:output:0(conv3d_26/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_26/ReluReluconv3d_26/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_21/Conv3D/ReadVariableOpReadVariableOp(conv3d_21_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_21/Conv3DConv3D!up_sampling3d_2/concat_2:output:0'conv3d_21/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_21/BiasAdd/ReadVariableOpReadVariableOp)conv3d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_21/BiasAddBiasAddconv3d_21/Conv3D:output:0(conv3d_21/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_21/ReluReluconv3d_21/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_16/Conv3DConv3Dup_sampling3d/concat_2:output:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_27/Conv3D/ReadVariableOpReadVariableOp(conv3d_27_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_27/Conv3DConv3Dconv3d_26/Relu:activations:0'conv3d_27/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_27/BiasAdd/ReadVariableOpReadVariableOp)conv3d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_27/BiasAddBiasAddconv3d_27/Conv3D:output:0(conv3d_27/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_27/ReluReluconv3d_27/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_22/Conv3D/ReadVariableOpReadVariableOp(conv3d_22_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_22/Conv3DConv3Dconv3d_21/Relu:activations:0'conv3d_22/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_22/BiasAdd/ReadVariableOpReadVariableOp)conv3d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_22/BiasAddBiasAddconv3d_22/Conv3D:output:0(conv3d_22/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_22/ReluReluconv3d_22/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_17/Conv3DConv3Dconv3d_16/Relu:activations:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_17/ReluReluconv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:���������a
up_sampling3d_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_5/splitSplit(up_sampling3d_5/split/split_dim:output:0conv3d_27/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_5/concatConcatV2up_sampling3d_5/split:output:0up_sampling3d_5/split:output:0up_sampling3d_5/split:output:1up_sampling3d_5/split:output:1up_sampling3d_5/split:output:2up_sampling3d_5/split:output:2up_sampling3d_5/split:output:3up_sampling3d_5/split:output:3up_sampling3d_5/split:output:4up_sampling3d_5/split:output:4up_sampling3d_5/split:output:5up_sampling3d_5/split:output:5up_sampling3d_5/split:output:6up_sampling3d_5/split:output:6up_sampling3d_5/split:output:7up_sampling3d_5/split:output:7up_sampling3d_5/split:output:8up_sampling3d_5/split:output:8up_sampling3d_5/split:output:9up_sampling3d_5/split:output:9up_sampling3d_5/split:output:10up_sampling3d_5/split:output:10up_sampling3d_5/split:output:11up_sampling3d_5/split:output:11up_sampling3d_5/split:output:12up_sampling3d_5/split:output:12up_sampling3d_5/split:output:13up_sampling3d_5/split:output:13up_sampling3d_5/split:output:14up_sampling3d_5/split:output:14up_sampling3d_5/split:output:15up_sampling3d_5/split:output:15$up_sampling3d_5/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� c
!up_sampling3d_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_5/split_1Split*up_sampling3d_5/split_1/split_dim:output:0up_sampling3d_5/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_split_
up_sampling3d_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_5/concat_1ConcatV2 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:9 up_sampling3d_5/split_1:output:9!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:15!up_sampling3d_5/split_1:output:15&up_sampling3d_5/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  c
!up_sampling3d_5/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_5/split_2Split*up_sampling3d_5/split_2/split_dim:output:0!up_sampling3d_5/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_split_
up_sampling3d_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_5/concat_2ConcatV2 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:9 up_sampling3d_5/split_2:output:9!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:15!up_sampling3d_5/split_2:output:15&up_sampling3d_5/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   a
up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_3/splitSplit(up_sampling3d_3/split/split_dim:output:0conv3d_22/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_3/concatConcatV2up_sampling3d_3/split:output:0up_sampling3d_3/split:output:0up_sampling3d_3/split:output:1up_sampling3d_3/split:output:1up_sampling3d_3/split:output:2up_sampling3d_3/split:output:2up_sampling3d_3/split:output:3up_sampling3d_3/split:output:3up_sampling3d_3/split:output:4up_sampling3d_3/split:output:4up_sampling3d_3/split:output:5up_sampling3d_3/split:output:5up_sampling3d_3/split:output:6up_sampling3d_3/split:output:6up_sampling3d_3/split:output:7up_sampling3d_3/split:output:7up_sampling3d_3/split:output:8up_sampling3d_3/split:output:8up_sampling3d_3/split:output:9up_sampling3d_3/split:output:9up_sampling3d_3/split:output:10up_sampling3d_3/split:output:10up_sampling3d_3/split:output:11up_sampling3d_3/split:output:11up_sampling3d_3/split:output:12up_sampling3d_3/split:output:12up_sampling3d_3/split:output:13up_sampling3d_3/split:output:13up_sampling3d_3/split:output:14up_sampling3d_3/split:output:14up_sampling3d_3/split:output:15up_sampling3d_3/split:output:15$up_sampling3d_3/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� c
!up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_3/split_1Split*up_sampling3d_3/split_1/split_dim:output:0up_sampling3d_3/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_split_
up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_3/concat_1ConcatV2 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:9 up_sampling3d_3/split_1:output:9!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:15&up_sampling3d_3/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  c
!up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_3/split_2Split*up_sampling3d_3/split_2/split_dim:output:0!up_sampling3d_3/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_split_
up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_3/concat_2ConcatV2 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:9 up_sampling3d_3/split_2:output:9!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:15&up_sampling3d_3/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   a
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0conv3d_17/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15$up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� c
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_split_
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15&up_sampling3d_1/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  c
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_split_
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15&up_sampling3d_1/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   �
conv3d_28/Conv3D/ReadVariableOpReadVariableOp(conv3d_28_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_28/Conv3DConv3D!up_sampling3d_5/concat_2:output:0'conv3d_28/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_28/BiasAdd/ReadVariableOpReadVariableOp)conv3d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_28/BiasAddBiasAddconv3d_28/Conv3D:output:0(conv3d_28/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_28/ReluReluconv3d_28/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_23/Conv3D/ReadVariableOpReadVariableOp(conv3d_23_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_23/Conv3DConv3D!up_sampling3d_3/concat_2:output:0'conv3d_23/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_23/BiasAdd/ReadVariableOpReadVariableOp)conv3d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_23/BiasAddBiasAddconv3d_23/Conv3D:output:0(conv3d_23/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_23/ReluReluconv3d_23/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_18/Conv3D/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_18/Conv3DConv3D!up_sampling3d_1/concat_2:output:0'conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_18/BiasAdd/ReadVariableOpReadVariableOp)conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_18/BiasAddBiasAddconv3d_18/Conv3D:output:0(conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_18/ReluReluconv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_19/Conv3D/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_19/Conv3DConv3Dconv3d_18/Relu:activations:0'conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_19/BiasAdd/ReadVariableOpReadVariableOp)conv3d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_19/BiasAddBiasAddconv3d_19/Conv3D:output:0(conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_19/ReluReluconv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_24/Conv3D/ReadVariableOpReadVariableOp(conv3d_24_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_24/Conv3DConv3Dconv3d_23/Relu:activations:0'conv3d_24/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_24/BiasAdd/ReadVariableOpReadVariableOp)conv3d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_24/BiasAddBiasAddconv3d_24/Conv3D:output:0(conv3d_24/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_24/ReluReluconv3d_24/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_29/Conv3D/ReadVariableOpReadVariableOp(conv3d_29_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_29/Conv3DConv3Dconv3d_28/Relu:activations:0'conv3d_29/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_29/BiasAdd/ReadVariableOpReadVariableOp)conv3d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_29/BiasAddBiasAddconv3d_29/Conv3D:output:0(conv3d_29/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_29/ReluReluconv3d_29/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
	add_1/addAddV2conv3d_19/Relu:activations:0conv3d_24/Relu:activations:0*
T0*3
_output_shapes!
:���������   
add_1/add_1AddV2add_1/add:z:0conv3d_29/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
conv3d_30/Conv3D/ReadVariableOpReadVariableOp(conv3d_30_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_30/Conv3DConv3Dadd_1/add_1:z:0'conv3d_30/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_30/BiasAdd/ReadVariableOpReadVariableOp)conv3d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_30/BiasAddBiasAddconv3d_30/Conv3D:output:0(conv3d_30/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   u
IdentityIdentityconv3d_30/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   �
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp!^conv3d_18/BiasAdd/ReadVariableOp ^conv3d_18/Conv3D/ReadVariableOp!^conv3d_19/BiasAdd/ReadVariableOp ^conv3d_19/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp!^conv3d_20/BiasAdd/ReadVariableOp ^conv3d_20/Conv3D/ReadVariableOp!^conv3d_21/BiasAdd/ReadVariableOp ^conv3d_21/Conv3D/ReadVariableOp!^conv3d_22/BiasAdd/ReadVariableOp ^conv3d_22/Conv3D/ReadVariableOp!^conv3d_23/BiasAdd/ReadVariableOp ^conv3d_23/Conv3D/ReadVariableOp!^conv3d_24/BiasAdd/ReadVariableOp ^conv3d_24/Conv3D/ReadVariableOp!^conv3d_25/BiasAdd/ReadVariableOp ^conv3d_25/Conv3D/ReadVariableOp!^conv3d_26/BiasAdd/ReadVariableOp ^conv3d_26/Conv3D/ReadVariableOp!^conv3d_27/BiasAdd/ReadVariableOp ^conv3d_27/Conv3D/ReadVariableOp!^conv3d_28/BiasAdd/ReadVariableOp ^conv3d_28/Conv3D/ReadVariableOp!^conv3d_29/BiasAdd/ReadVariableOp ^conv3d_29/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp!^conv3d_30/BiasAdd/ReadVariableOp ^conv3d_30/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
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
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�

�
D__inference_conv3d_30_layer_call_and_return_conditional_losses_96234

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_95444

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_91404

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_28_layer_call_fn_96129

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_28_layer_call_and_return_conditional_losses_92110{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�	
^
B__inference_reshape_layer_call_and_return_conditional_losses_91621

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
valueB:�
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
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_17_layer_call_and_return_conditional_losses_95842

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_24_layer_call_and_return_conditional_losses_92178

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91298

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_9_layer_call_fn_95493

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_91576{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
I
-__inference_up_sampling3d_layer_call_fn_95641

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_91806l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�%
f
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_92034

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_15_layer_call_and_return_conditional_losses_95596

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_up_sampling3d_5_layer_call_fn_96019

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_91971l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_24_layer_call_and_return_conditional_losses_96180

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
d
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_91806

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
I
-__inference_max_pooling3d_layer_call_fn_95234

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91426l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91414

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
ߩ
�0
@__inference_model_layer_call_and_return_conditional_losses_95104

inputsF
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
identity��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp� conv3d_10/BiasAdd/ReadVariableOp�conv3d_10/Conv3D/ReadVariableOp� conv3d_11/BiasAdd/ReadVariableOp�conv3d_11/Conv3D/ReadVariableOp� conv3d_12/BiasAdd/ReadVariableOp�conv3d_12/Conv3D/ReadVariableOp� conv3d_13/BiasAdd/ReadVariableOp�conv3d_13/Conv3D/ReadVariableOp� conv3d_14/BiasAdd/ReadVariableOp�conv3d_14/Conv3D/ReadVariableOp� conv3d_15/BiasAdd/ReadVariableOp�conv3d_15/Conv3D/ReadVariableOp� conv3d_16/BiasAdd/ReadVariableOp�conv3d_16/Conv3D/ReadVariableOp� conv3d_17/BiasAdd/ReadVariableOp�conv3d_17/Conv3D/ReadVariableOp� conv3d_18/BiasAdd/ReadVariableOp�conv3d_18/Conv3D/ReadVariableOp� conv3d_19/BiasAdd/ReadVariableOp�conv3d_19/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp� conv3d_20/BiasAdd/ReadVariableOp�conv3d_20/Conv3D/ReadVariableOp� conv3d_21/BiasAdd/ReadVariableOp�conv3d_21/Conv3D/ReadVariableOp� conv3d_22/BiasAdd/ReadVariableOp�conv3d_22/Conv3D/ReadVariableOp� conv3d_23/BiasAdd/ReadVariableOp�conv3d_23/Conv3D/ReadVariableOp� conv3d_24/BiasAdd/ReadVariableOp�conv3d_24/Conv3D/ReadVariableOp� conv3d_25/BiasAdd/ReadVariableOp�conv3d_25/Conv3D/ReadVariableOp� conv3d_26/BiasAdd/ReadVariableOp�conv3d_26/Conv3D/ReadVariableOp� conv3d_27/BiasAdd/ReadVariableOp�conv3d_27/Conv3D/ReadVariableOp� conv3d_28/BiasAdd/ReadVariableOp�conv3d_28/Conv3D/ReadVariableOp� conv3d_29/BiasAdd/ReadVariableOp�conv3d_29/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp� conv3d_30/BiasAdd/ReadVariableOp�conv3d_30/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�conv3d_8/BiasAdd/ReadVariableOp�conv3d_8/Conv3D/ReadVariableOp�conv3d_9/BiasAdd/ReadVariableOp�conv3d_9/Conv3D/ReadVariableOp�
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_10/Conv3DConv3Dinputs'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_10/ReluReluconv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_5/Conv3DConv3Dinputs&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_11/Conv3DConv3Dconv3d_10/Relu:activations:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_11/ReluReluconv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_6/Conv3DConv3Dconv3d_5/Relu:activations:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
max_pooling3d_4/MaxPool3D	MaxPool3Dconv3d_11/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_12/Conv3DConv3D"max_pooling3d_4/MaxPool3D:output:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_12/ReluReluconv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_7/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_13/Conv3DConv3Dconv3d_12/Relu:activations:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_13/ReluReluconv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_8/Conv3DConv3Dconv3d_7/Relu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_8/ReluReluconv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
max_pooling3d_5/MaxPool3D	MaxPool3Dconv3d_13/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_4/TanhTanhconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_9/Conv3DConv3D"max_pooling3d_3/MaxPool3D:output:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������n
conv3d_9/TanhTanhconv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_14/Conv3DConv3D"max_pooling3d_5/MaxPool3D:output:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_14/TanhTanhconv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:���������t
add/addAddV2conv3d_4/Tanh:y:0conv3d_9/Tanh:y:0*
T0*3
_output_shapes!
:���������q
	add/add_1AddV2add/add:z:0conv3d_14/Tanh:y:0*
T0*3
_output_shapes!
:���������J
reshape/ShapeShapeadd/add_1:z:0*
T0*
_output_shapes
:e
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
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:|
reshape/ReshapeReshapeadd/add_1:z:0reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:����������W
reshape_1/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:g
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
valueB:�
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
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapereshape/Reshape:output:0 reshape_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:����������
conv3d_25/Conv3D/ReadVariableOpReadVariableOp(conv3d_25_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_25/Conv3DConv3Dreshape_1/Reshape:output:0'conv3d_25/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_25/BiasAdd/ReadVariableOpReadVariableOp)conv3d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_25/BiasAddBiasAddconv3d_25/Conv3D:output:0(conv3d_25/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_25/ReluReluconv3d_25/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_20/Conv3D/ReadVariableOpReadVariableOp(conv3d_20_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_20/Conv3DConv3Dreshape_1/Reshape:output:0'conv3d_20/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_20/BiasAdd/ReadVariableOpReadVariableOp)conv3d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_20/BiasAddBiasAddconv3d_20/Conv3D:output:0(conv3d_20/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_20/ReluReluconv3d_20/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_15/Conv3DConv3Dreshape_1/Reshape:output:0'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:���������a
up_sampling3d_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/splitSplit(up_sampling3d_4/split/split_dim:output:0conv3d_25/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/concatConcatV2up_sampling3d_4/split:output:0up_sampling3d_4/split:output:0up_sampling3d_4/split:output:1up_sampling3d_4/split:output:1up_sampling3d_4/split:output:2up_sampling3d_4/split:output:2up_sampling3d_4/split:output:3up_sampling3d_4/split:output:3up_sampling3d_4/split:output:4up_sampling3d_4/split:output:4up_sampling3d_4/split:output:5up_sampling3d_4/split:output:5up_sampling3d_4/split:output:6up_sampling3d_4/split:output:6up_sampling3d_4/split:output:7up_sampling3d_4/split:output:7$up_sampling3d_4/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/split_1Split*up_sampling3d_4/split_1/split_dim:output:0up_sampling3d_4/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/concat_1ConcatV2 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:0 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:1 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:2 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:3 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:4 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:5 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:6 up_sampling3d_4/split_1:output:7 up_sampling3d_4/split_1:output:7&up_sampling3d_4/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_4/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/split_2Split*up_sampling3d_4/split_2/split_dim:output:0!up_sampling3d_4/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_4/concat_2ConcatV2 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:0 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:1 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:2 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:3 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:4 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:5 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:6 up_sampling3d_4/split_2:output:7 up_sampling3d_4/split_2:output:7&up_sampling3d_4/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������a
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0conv3d_20/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7$up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7&up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������c
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split_
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7&up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������_
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0conv3d_15/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split[
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7"up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������a
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7$up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������a
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7$up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:����������
conv3d_26/Conv3D/ReadVariableOpReadVariableOp(conv3d_26_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_26/Conv3DConv3D!up_sampling3d_4/concat_2:output:0'conv3d_26/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_26/BiasAdd/ReadVariableOpReadVariableOp)conv3d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_26/BiasAddBiasAddconv3d_26/Conv3D:output:0(conv3d_26/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_26/ReluReluconv3d_26/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_21/Conv3D/ReadVariableOpReadVariableOp(conv3d_21_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_21/Conv3DConv3D!up_sampling3d_2/concat_2:output:0'conv3d_21/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_21/BiasAdd/ReadVariableOpReadVariableOp)conv3d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_21/BiasAddBiasAddconv3d_21/Conv3D:output:0(conv3d_21/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_21/ReluReluconv3d_21/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_16/Conv3DConv3Dup_sampling3d/concat_2:output:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_27/Conv3D/ReadVariableOpReadVariableOp(conv3d_27_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_27/Conv3DConv3Dconv3d_26/Relu:activations:0'conv3d_27/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_27/BiasAdd/ReadVariableOpReadVariableOp)conv3d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_27/BiasAddBiasAddconv3d_27/Conv3D:output:0(conv3d_27/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_27/ReluReluconv3d_27/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_22/Conv3D/ReadVariableOpReadVariableOp(conv3d_22_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_22/Conv3DConv3Dconv3d_21/Relu:activations:0'conv3d_22/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_22/BiasAdd/ReadVariableOpReadVariableOp)conv3d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_22/BiasAddBiasAddconv3d_22/Conv3D:output:0(conv3d_22/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_22/ReluReluconv3d_22/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_17/Conv3DConv3Dconv3d_16/Relu:activations:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������p
conv3d_17/ReluReluconv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:���������a
up_sampling3d_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_5/splitSplit(up_sampling3d_5/split/split_dim:output:0conv3d_27/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_5/concatConcatV2up_sampling3d_5/split:output:0up_sampling3d_5/split:output:0up_sampling3d_5/split:output:1up_sampling3d_5/split:output:1up_sampling3d_5/split:output:2up_sampling3d_5/split:output:2up_sampling3d_5/split:output:3up_sampling3d_5/split:output:3up_sampling3d_5/split:output:4up_sampling3d_5/split:output:4up_sampling3d_5/split:output:5up_sampling3d_5/split:output:5up_sampling3d_5/split:output:6up_sampling3d_5/split:output:6up_sampling3d_5/split:output:7up_sampling3d_5/split:output:7up_sampling3d_5/split:output:8up_sampling3d_5/split:output:8up_sampling3d_5/split:output:9up_sampling3d_5/split:output:9up_sampling3d_5/split:output:10up_sampling3d_5/split:output:10up_sampling3d_5/split:output:11up_sampling3d_5/split:output:11up_sampling3d_5/split:output:12up_sampling3d_5/split:output:12up_sampling3d_5/split:output:13up_sampling3d_5/split:output:13up_sampling3d_5/split:output:14up_sampling3d_5/split:output:14up_sampling3d_5/split:output:15up_sampling3d_5/split:output:15$up_sampling3d_5/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� c
!up_sampling3d_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_5/split_1Split*up_sampling3d_5/split_1/split_dim:output:0up_sampling3d_5/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_split_
up_sampling3d_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_5/concat_1ConcatV2 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:0 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:1 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:2 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:3 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:4 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:5 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:6 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:7 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:8 up_sampling3d_5/split_1:output:9 up_sampling3d_5/split_1:output:9!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:10!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:11!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:12!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:13!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:14!up_sampling3d_5/split_1:output:15!up_sampling3d_5/split_1:output:15&up_sampling3d_5/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  c
!up_sampling3d_5/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_5/split_2Split*up_sampling3d_5/split_2/split_dim:output:0!up_sampling3d_5/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_split_
up_sampling3d_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_5/concat_2ConcatV2 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:0 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:1 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:2 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:3 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:4 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:5 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:6 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:7 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:8 up_sampling3d_5/split_2:output:9 up_sampling3d_5/split_2:output:9!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:10!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:11!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:12!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:13!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:14!up_sampling3d_5/split_2:output:15!up_sampling3d_5/split_2:output:15&up_sampling3d_5/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   a
up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_3/splitSplit(up_sampling3d_3/split/split_dim:output:0conv3d_22/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_3/concatConcatV2up_sampling3d_3/split:output:0up_sampling3d_3/split:output:0up_sampling3d_3/split:output:1up_sampling3d_3/split:output:1up_sampling3d_3/split:output:2up_sampling3d_3/split:output:2up_sampling3d_3/split:output:3up_sampling3d_3/split:output:3up_sampling3d_3/split:output:4up_sampling3d_3/split:output:4up_sampling3d_3/split:output:5up_sampling3d_3/split:output:5up_sampling3d_3/split:output:6up_sampling3d_3/split:output:6up_sampling3d_3/split:output:7up_sampling3d_3/split:output:7up_sampling3d_3/split:output:8up_sampling3d_3/split:output:8up_sampling3d_3/split:output:9up_sampling3d_3/split:output:9up_sampling3d_3/split:output:10up_sampling3d_3/split:output:10up_sampling3d_3/split:output:11up_sampling3d_3/split:output:11up_sampling3d_3/split:output:12up_sampling3d_3/split:output:12up_sampling3d_3/split:output:13up_sampling3d_3/split:output:13up_sampling3d_3/split:output:14up_sampling3d_3/split:output:14up_sampling3d_3/split:output:15up_sampling3d_3/split:output:15$up_sampling3d_3/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� c
!up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_3/split_1Split*up_sampling3d_3/split_1/split_dim:output:0up_sampling3d_3/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_split_
up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_3/concat_1ConcatV2 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:9 up_sampling3d_3/split_1:output:9!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:15&up_sampling3d_3/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  c
!up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_3/split_2Split*up_sampling3d_3/split_2/split_dim:output:0!up_sampling3d_3/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_split_
up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_3/concat_2ConcatV2 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:9 up_sampling3d_3/split_2:output:9!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:15&up_sampling3d_3/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   a
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0conv3d_17/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_split]
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15$up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� c
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_split_
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15&up_sampling3d_1/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  c
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_split_
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�	
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15&up_sampling3d_1/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   �
conv3d_28/Conv3D/ReadVariableOpReadVariableOp(conv3d_28_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_28/Conv3DConv3D!up_sampling3d_5/concat_2:output:0'conv3d_28/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_28/BiasAdd/ReadVariableOpReadVariableOp)conv3d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_28/BiasAddBiasAddconv3d_28/Conv3D:output:0(conv3d_28/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_28/ReluReluconv3d_28/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_23/Conv3D/ReadVariableOpReadVariableOp(conv3d_23_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_23/Conv3DConv3D!up_sampling3d_3/concat_2:output:0'conv3d_23/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_23/BiasAdd/ReadVariableOpReadVariableOp)conv3d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_23/BiasAddBiasAddconv3d_23/Conv3D:output:0(conv3d_23/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_23/ReluReluconv3d_23/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_18/Conv3D/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_18/Conv3DConv3D!up_sampling3d_1/concat_2:output:0'conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_18/BiasAdd/ReadVariableOpReadVariableOp)conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_18/BiasAddBiasAddconv3d_18/Conv3D:output:0(conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_18/ReluReluconv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_19/Conv3D/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_19/Conv3DConv3Dconv3d_18/Relu:activations:0'conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_19/BiasAdd/ReadVariableOpReadVariableOp)conv3d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_19/BiasAddBiasAddconv3d_19/Conv3D:output:0(conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_19/ReluReluconv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_24/Conv3D/ReadVariableOpReadVariableOp(conv3d_24_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_24/Conv3DConv3Dconv3d_23/Relu:activations:0'conv3d_24/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_24/BiasAdd/ReadVariableOpReadVariableOp)conv3d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_24/BiasAddBiasAddconv3d_24/Conv3D:output:0(conv3d_24/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_24/ReluReluconv3d_24/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_29/Conv3D/ReadVariableOpReadVariableOp(conv3d_29_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_29/Conv3DConv3Dconv3d_28/Relu:activations:0'conv3d_29/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_29/BiasAdd/ReadVariableOpReadVariableOp)conv3d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_29/BiasAddBiasAddconv3d_29/Conv3D:output:0(conv3d_29/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_29/ReluReluconv3d_29/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
	add_1/addAddV2conv3d_19/Relu:activations:0conv3d_24/Relu:activations:0*
T0*3
_output_shapes!
:���������   
add_1/add_1AddV2add_1/add:z:0conv3d_29/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
conv3d_30/Conv3D/ReadVariableOpReadVariableOp(conv3d_30_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_30/Conv3DConv3Dadd_1/add_1:z:0'conv3d_30/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_30/BiasAdd/ReadVariableOpReadVariableOp)conv3d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_30/BiasAddBiasAddconv3d_30/Conv3D:output:0(conv3d_30/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   u
IdentityIdentityconv3d_30/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   �
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp!^conv3d_18/BiasAdd/ReadVariableOp ^conv3d_18/Conv3D/ReadVariableOp!^conv3d_19/BiasAdd/ReadVariableOp ^conv3d_19/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp!^conv3d_20/BiasAdd/ReadVariableOp ^conv3d_20/Conv3D/ReadVariableOp!^conv3d_21/BiasAdd/ReadVariableOp ^conv3d_21/Conv3D/ReadVariableOp!^conv3d_22/BiasAdd/ReadVariableOp ^conv3d_22/Conv3D/ReadVariableOp!^conv3d_23/BiasAdd/ReadVariableOp ^conv3d_23/Conv3D/ReadVariableOp!^conv3d_24/BiasAdd/ReadVariableOp ^conv3d_24/Conv3D/ReadVariableOp!^conv3d_25/BiasAdd/ReadVariableOp ^conv3d_25/Conv3D/ReadVariableOp!^conv3d_26/BiasAdd/ReadVariableOp ^conv3d_26/Conv3D/ReadVariableOp!^conv3d_27/BiasAdd/ReadVariableOp ^conv3d_27/Conv3D/ReadVariableOp!^conv3d_28/BiasAdd/ReadVariableOp ^conv3d_28/Conv3D/ReadVariableOp!^conv3d_29/BiasAdd/ReadVariableOp ^conv3d_29/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp!^conv3d_30/BiasAdd/ReadVariableOp ^conv3d_30/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
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
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_3_layer_call_fn_95429

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91286�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_15_layer_call_fn_95585

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_91685{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
��
�5
 __inference__wrapped_model_91229
input_1L
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
identity��#model/conv3d/BiasAdd/ReadVariableOp�"model/conv3d/Conv3D/ReadVariableOp�%model/conv3d_1/BiasAdd/ReadVariableOp�$model/conv3d_1/Conv3D/ReadVariableOp�&model/conv3d_10/BiasAdd/ReadVariableOp�%model/conv3d_10/Conv3D/ReadVariableOp�&model/conv3d_11/BiasAdd/ReadVariableOp�%model/conv3d_11/Conv3D/ReadVariableOp�&model/conv3d_12/BiasAdd/ReadVariableOp�%model/conv3d_12/Conv3D/ReadVariableOp�&model/conv3d_13/BiasAdd/ReadVariableOp�%model/conv3d_13/Conv3D/ReadVariableOp�&model/conv3d_14/BiasAdd/ReadVariableOp�%model/conv3d_14/Conv3D/ReadVariableOp�&model/conv3d_15/BiasAdd/ReadVariableOp�%model/conv3d_15/Conv3D/ReadVariableOp�&model/conv3d_16/BiasAdd/ReadVariableOp�%model/conv3d_16/Conv3D/ReadVariableOp�&model/conv3d_17/BiasAdd/ReadVariableOp�%model/conv3d_17/Conv3D/ReadVariableOp�&model/conv3d_18/BiasAdd/ReadVariableOp�%model/conv3d_18/Conv3D/ReadVariableOp�&model/conv3d_19/BiasAdd/ReadVariableOp�%model/conv3d_19/Conv3D/ReadVariableOp�%model/conv3d_2/BiasAdd/ReadVariableOp�$model/conv3d_2/Conv3D/ReadVariableOp�&model/conv3d_20/BiasAdd/ReadVariableOp�%model/conv3d_20/Conv3D/ReadVariableOp�&model/conv3d_21/BiasAdd/ReadVariableOp�%model/conv3d_21/Conv3D/ReadVariableOp�&model/conv3d_22/BiasAdd/ReadVariableOp�%model/conv3d_22/Conv3D/ReadVariableOp�&model/conv3d_23/BiasAdd/ReadVariableOp�%model/conv3d_23/Conv3D/ReadVariableOp�&model/conv3d_24/BiasAdd/ReadVariableOp�%model/conv3d_24/Conv3D/ReadVariableOp�&model/conv3d_25/BiasAdd/ReadVariableOp�%model/conv3d_25/Conv3D/ReadVariableOp�&model/conv3d_26/BiasAdd/ReadVariableOp�%model/conv3d_26/Conv3D/ReadVariableOp�&model/conv3d_27/BiasAdd/ReadVariableOp�%model/conv3d_27/Conv3D/ReadVariableOp�&model/conv3d_28/BiasAdd/ReadVariableOp�%model/conv3d_28/Conv3D/ReadVariableOp�&model/conv3d_29/BiasAdd/ReadVariableOp�%model/conv3d_29/Conv3D/ReadVariableOp�%model/conv3d_3/BiasAdd/ReadVariableOp�$model/conv3d_3/Conv3D/ReadVariableOp�&model/conv3d_30/BiasAdd/ReadVariableOp�%model/conv3d_30/Conv3D/ReadVariableOp�%model/conv3d_4/BiasAdd/ReadVariableOp�$model/conv3d_4/Conv3D/ReadVariableOp�%model/conv3d_5/BiasAdd/ReadVariableOp�$model/conv3d_5/Conv3D/ReadVariableOp�%model/conv3d_6/BiasAdd/ReadVariableOp�$model/conv3d_6/Conv3D/ReadVariableOp�%model/conv3d_7/BiasAdd/ReadVariableOp�$model/conv3d_7/Conv3D/ReadVariableOp�%model/conv3d_8/BiasAdd/ReadVariableOp�$model/conv3d_8/Conv3D/ReadVariableOp�%model/conv3d_9/BiasAdd/ReadVariableOp�$model/conv3d_9/Conv3D/ReadVariableOp�
%model/conv3d_10/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_10/Conv3DConv3Dinput_1-model/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_10/BiasAddBiasAddmodel/conv3d_10/Conv3D:output:0.model/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_10/ReluRelu model/conv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_5/Conv3DConv3Dinput_1,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_11/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_11/Conv3DConv3D"model/conv3d_10/Relu:activations:0-model/conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_11/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_11/BiasAddBiasAddmodel/conv3d_11/Conv3D:output:0.model/conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_11/ReluRelu model/conv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_6/Conv3DConv3D!model/conv3d_5/Relu:activations:0,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
model/max_pooling3d_4/MaxPool3D	MaxPool3D"model/conv3d_11/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
model/max_pooling3d/MaxPool3D	MaxPool3D!model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
%model/conv3d_12/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_12/Conv3DConv3D(model/max_pooling3d_4/MaxPool3D:output:0-model/conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_12/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_12/BiasAddBiasAddmodel/conv3d_12/Conv3D:output:0.model/conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_12/ReluRelu model/conv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_7/Conv3DConv3D(model/max_pooling3d_2/MaxPool3D:output:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������z
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_2/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������z
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_13/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_13/Conv3DConv3D"model/conv3d_12/Relu:activations:0-model/conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_13/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_13/BiasAddBiasAddmodel/conv3d_13/Conv3D:output:0.model/conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_13/ReluRelu model/conv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
$model/conv3d_8/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_8/Conv3DConv3D!model/conv3d_7/Relu:activations:0,model/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
%model/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_8/BiasAddBiasAddmodel/conv3d_8/Conv3D:output:0-model/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������z
model/conv3d_8/ReluRelumodel/conv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������z
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
model/max_pooling3d_5/MaxPool3D	MaxPool3D"model/conv3d_13/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
model/max_pooling3d_3/MaxPool3D	MaxPool3D!model/conv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
�
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_4/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������z
model/conv3d_4/TanhTanhmodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
$model/conv3d_9/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_9/Conv3DConv3D(model/max_pooling3d_3/MaxPool3D:output:0,model/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
%model/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_9/BiasAddBiasAddmodel/conv3d_9/Conv3D:output:0-model/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������z
model/conv3d_9/TanhTanhmodel/conv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_14/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_14/Conv3DConv3D(model/max_pooling3d_5/MaxPool3D:output:0-model/conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_14/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_14/BiasAddBiasAddmodel/conv3d_14/Conv3D:output:0.model/conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_14/TanhTanh model/conv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
model/add/addAddV2model/conv3d_4/Tanh:y:0model/conv3d_9/Tanh:y:0*
T0*3
_output_shapes!
:����������
model/add/add_1AddV2model/add/add:z:0model/conv3d_14/Tanh:y:0*
T0*3
_output_shapes!
:���������V
model/reshape/ShapeShapemodel/add/add_1:z:0*
T0*
_output_shapes
:k
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
valueB:�
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/reshape/ReshapeReshapemodel/add/add_1:z:0$model/reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:����������c
model/reshape_1/ShapeShapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
:m
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
valueB:�
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
value	B :a
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0(model/reshape_1/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:�
model/reshape_1/ReshapeReshapemodel/reshape/Reshape:output:0&model/reshape_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_25/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_25_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_25/Conv3DConv3D model/reshape_1/Reshape:output:0-model/conv3d_25/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_25/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_25/BiasAddBiasAddmodel/conv3d_25/Conv3D:output:0.model/conv3d_25/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_25/ReluRelu model/conv3d_25/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_20/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_20_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_20/Conv3DConv3D model/reshape_1/Reshape:output:0-model/conv3d_20/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_20/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_20/BiasAddBiasAddmodel/conv3d_20/Conv3D:output:0.model/conv3d_20/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_20/ReluRelu model/conv3d_20/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_15/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_15/Conv3DConv3D model/reshape_1/Reshape:output:0-model/conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_15/BiasAddBiasAddmodel/conv3d_15/Conv3D:output:0.model/conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_15/ReluRelu model/conv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:���������g
%model/up_sampling3d_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_4/splitSplit.model/up_sampling3d_4/split/split_dim:output:0"model/conv3d_25/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_4/concatConcatV2$model/up_sampling3d_4/split:output:0$model/up_sampling3d_4/split:output:0$model/up_sampling3d_4/split:output:1$model/up_sampling3d_4/split:output:1$model/up_sampling3d_4/split:output:2$model/up_sampling3d_4/split:output:2$model/up_sampling3d_4/split:output:3$model/up_sampling3d_4/split:output:3$model/up_sampling3d_4/split:output:4$model/up_sampling3d_4/split:output:4$model/up_sampling3d_4/split:output:5$model/up_sampling3d_4/split:output:5$model/up_sampling3d_4/split:output:6$model/up_sampling3d_4/split:output:6$model/up_sampling3d_4/split:output:7$model/up_sampling3d_4/split:output:7*model/up_sampling3d_4/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������i
'model/up_sampling3d_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_4/split_1Split0model/up_sampling3d_4/split_1/split_dim:output:0%model/up_sampling3d_4/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splite
#model/up_sampling3d_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_4/concat_1ConcatV2&model/up_sampling3d_4/split_1:output:0&model/up_sampling3d_4/split_1:output:0&model/up_sampling3d_4/split_1:output:1&model/up_sampling3d_4/split_1:output:1&model/up_sampling3d_4/split_1:output:2&model/up_sampling3d_4/split_1:output:2&model/up_sampling3d_4/split_1:output:3&model/up_sampling3d_4/split_1:output:3&model/up_sampling3d_4/split_1:output:4&model/up_sampling3d_4/split_1:output:4&model/up_sampling3d_4/split_1:output:5&model/up_sampling3d_4/split_1:output:5&model/up_sampling3d_4/split_1:output:6&model/up_sampling3d_4/split_1:output:6&model/up_sampling3d_4/split_1:output:7&model/up_sampling3d_4/split_1:output:7,model/up_sampling3d_4/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������i
'model/up_sampling3d_4/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_4/split_2Split0model/up_sampling3d_4/split_2/split_dim:output:0'model/up_sampling3d_4/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splite
#model/up_sampling3d_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_4/concat_2ConcatV2&model/up_sampling3d_4/split_2:output:0&model/up_sampling3d_4/split_2:output:0&model/up_sampling3d_4/split_2:output:1&model/up_sampling3d_4/split_2:output:1&model/up_sampling3d_4/split_2:output:2&model/up_sampling3d_4/split_2:output:2&model/up_sampling3d_4/split_2:output:3&model/up_sampling3d_4/split_2:output:3&model/up_sampling3d_4/split_2:output:4&model/up_sampling3d_4/split_2:output:4&model/up_sampling3d_4/split_2:output:5&model/up_sampling3d_4/split_2:output:5&model/up_sampling3d_4/split_2:output:6&model/up_sampling3d_4/split_2:output:6&model/up_sampling3d_4/split_2:output:7&model/up_sampling3d_4/split_2:output:7,model/up_sampling3d_4/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������g
%model/up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_2/splitSplit.model/up_sampling3d_2/split/split_dim:output:0"model/conv3d_20/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_2/concatConcatV2$model/up_sampling3d_2/split:output:0$model/up_sampling3d_2/split:output:0$model/up_sampling3d_2/split:output:1$model/up_sampling3d_2/split:output:1$model/up_sampling3d_2/split:output:2$model/up_sampling3d_2/split:output:2$model/up_sampling3d_2/split:output:3$model/up_sampling3d_2/split:output:3$model/up_sampling3d_2/split:output:4$model/up_sampling3d_2/split:output:4$model/up_sampling3d_2/split:output:5$model/up_sampling3d_2/split:output:5$model/up_sampling3d_2/split:output:6$model/up_sampling3d_2/split:output:6$model/up_sampling3d_2/split:output:7$model/up_sampling3d_2/split:output:7*model/up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������i
'model/up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_2/split_1Split0model/up_sampling3d_2/split_1/split_dim:output:0%model/up_sampling3d_2/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splite
#model/up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_2/concat_1ConcatV2&model/up_sampling3d_2/split_1:output:0&model/up_sampling3d_2/split_1:output:0&model/up_sampling3d_2/split_1:output:1&model/up_sampling3d_2/split_1:output:1&model/up_sampling3d_2/split_1:output:2&model/up_sampling3d_2/split_1:output:2&model/up_sampling3d_2/split_1:output:3&model/up_sampling3d_2/split_1:output:3&model/up_sampling3d_2/split_1:output:4&model/up_sampling3d_2/split_1:output:4&model/up_sampling3d_2/split_1:output:5&model/up_sampling3d_2/split_1:output:5&model/up_sampling3d_2/split_1:output:6&model/up_sampling3d_2/split_1:output:6&model/up_sampling3d_2/split_1:output:7&model/up_sampling3d_2/split_1:output:7,model/up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������i
'model/up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_2/split_2Split0model/up_sampling3d_2/split_2/split_dim:output:0'model/up_sampling3d_2/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splite
#model/up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_2/concat_2ConcatV2&model/up_sampling3d_2/split_2:output:0&model/up_sampling3d_2/split_2:output:0&model/up_sampling3d_2/split_2:output:1&model/up_sampling3d_2/split_2:output:1&model/up_sampling3d_2/split_2:output:2&model/up_sampling3d_2/split_2:output:2&model/up_sampling3d_2/split_2:output:3&model/up_sampling3d_2/split_2:output:3&model/up_sampling3d_2/split_2:output:4&model/up_sampling3d_2/split_2:output:4&model/up_sampling3d_2/split_2:output:5&model/up_sampling3d_2/split_2:output:5&model/up_sampling3d_2/split_2:output:6&model/up_sampling3d_2/split_2:output:6&model/up_sampling3d_2/split_2:output:7&model/up_sampling3d_2/split_2:output:7,model/up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
#model/up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d/splitSplit,model/up_sampling3d/split/split_dim:output:0"model/conv3d_15/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splita
model/up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d/concatConcatV2"model/up_sampling3d/split:output:0"model/up_sampling3d/split:output:0"model/up_sampling3d/split:output:1"model/up_sampling3d/split:output:1"model/up_sampling3d/split:output:2"model/up_sampling3d/split:output:2"model/up_sampling3d/split:output:3"model/up_sampling3d/split:output:3"model/up_sampling3d/split:output:4"model/up_sampling3d/split:output:4"model/up_sampling3d/split:output:5"model/up_sampling3d/split:output:5"model/up_sampling3d/split:output:6"model/up_sampling3d/split:output:6"model/up_sampling3d/split:output:7"model/up_sampling3d/split:output:7(model/up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������g
%model/up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d/split_1Split.model/up_sampling3d/split_1/split_dim:output:0#model/up_sampling3d/concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d/concat_1ConcatV2$model/up_sampling3d/split_1:output:0$model/up_sampling3d/split_1:output:0$model/up_sampling3d/split_1:output:1$model/up_sampling3d/split_1:output:1$model/up_sampling3d/split_1:output:2$model/up_sampling3d/split_1:output:2$model/up_sampling3d/split_1:output:3$model/up_sampling3d/split_1:output:3$model/up_sampling3d/split_1:output:4$model/up_sampling3d/split_1:output:4$model/up_sampling3d/split_1:output:5$model/up_sampling3d/split_1:output:5$model/up_sampling3d/split_1:output:6$model/up_sampling3d/split_1:output:6$model/up_sampling3d/split_1:output:7$model/up_sampling3d/split_1:output:7*model/up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������g
%model/up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d/split_2Split.model/up_sampling3d/split_2/split_dim:output:0%model/up_sampling3d/concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d/concat_2ConcatV2$model/up_sampling3d/split_2:output:0$model/up_sampling3d/split_2:output:0$model/up_sampling3d/split_2:output:1$model/up_sampling3d/split_2:output:1$model/up_sampling3d/split_2:output:2$model/up_sampling3d/split_2:output:2$model/up_sampling3d/split_2:output:3$model/up_sampling3d/split_2:output:3$model/up_sampling3d/split_2:output:4$model/up_sampling3d/split_2:output:4$model/up_sampling3d/split_2:output:5$model/up_sampling3d/split_2:output:5$model/up_sampling3d/split_2:output:6$model/up_sampling3d/split_2:output:6$model/up_sampling3d/split_2:output:7$model/up_sampling3d/split_2:output:7*model/up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:����������
%model/conv3d_26/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_26_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_26/Conv3DConv3D'model/up_sampling3d_4/concat_2:output:0-model/conv3d_26/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_26/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_26/BiasAddBiasAddmodel/conv3d_26/Conv3D:output:0.model/conv3d_26/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_26/ReluRelu model/conv3d_26/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_21/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_21_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_21/Conv3DConv3D'model/up_sampling3d_2/concat_2:output:0-model/conv3d_21/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_21/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_21/BiasAddBiasAddmodel/conv3d_21/Conv3D:output:0.model/conv3d_21/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_21/ReluRelu model/conv3d_21/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_16/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_16/Conv3DConv3D%model/up_sampling3d/concat_2:output:0-model/conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_16/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_16/BiasAddBiasAddmodel/conv3d_16/Conv3D:output:0.model/conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_16/ReluRelu model/conv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_27/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_27_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_27/Conv3DConv3D"model/conv3d_26/Relu:activations:0-model/conv3d_27/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_27/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_27/BiasAddBiasAddmodel/conv3d_27/Conv3D:output:0.model/conv3d_27/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_27/ReluRelu model/conv3d_27/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_22/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_22_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_22/Conv3DConv3D"model/conv3d_21/Relu:activations:0-model/conv3d_22/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_22/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_22/BiasAddBiasAddmodel/conv3d_22/Conv3D:output:0.model/conv3d_22/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_22/ReluRelu model/conv3d_22/BiasAdd:output:0*
T0*3
_output_shapes!
:����������
%model/conv3d_17/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_17/Conv3DConv3D"model/conv3d_16/Relu:activations:0-model/conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
�
&model/conv3d_17/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_17/BiasAddBiasAddmodel/conv3d_17/Conv3D:output:0.model/conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������|
model/conv3d_17/ReluRelu model/conv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:���������g
%model/up_sampling3d_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_5/splitSplit.model/up_sampling3d_5/split/split_dim:output:0"model/conv3d_27/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�

model/up_sampling3d_5/concatConcatV2$model/up_sampling3d_5/split:output:0$model/up_sampling3d_5/split:output:0$model/up_sampling3d_5/split:output:1$model/up_sampling3d_5/split:output:1$model/up_sampling3d_5/split:output:2$model/up_sampling3d_5/split:output:2$model/up_sampling3d_5/split:output:3$model/up_sampling3d_5/split:output:3$model/up_sampling3d_5/split:output:4$model/up_sampling3d_5/split:output:4$model/up_sampling3d_5/split:output:5$model/up_sampling3d_5/split:output:5$model/up_sampling3d_5/split:output:6$model/up_sampling3d_5/split:output:6$model/up_sampling3d_5/split:output:7$model/up_sampling3d_5/split:output:7$model/up_sampling3d_5/split:output:8$model/up_sampling3d_5/split:output:8$model/up_sampling3d_5/split:output:9$model/up_sampling3d_5/split:output:9%model/up_sampling3d_5/split:output:10%model/up_sampling3d_5/split:output:10%model/up_sampling3d_5/split:output:11%model/up_sampling3d_5/split:output:11%model/up_sampling3d_5/split:output:12%model/up_sampling3d_5/split:output:12%model/up_sampling3d_5/split:output:13%model/up_sampling3d_5/split:output:13%model/up_sampling3d_5/split:output:14%model/up_sampling3d_5/split:output:14%model/up_sampling3d_5/split:output:15%model/up_sampling3d_5/split:output:15*model/up_sampling3d_5/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� i
'model/up_sampling3d_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_5/split_1Split0model/up_sampling3d_5/split_1/split_dim:output:0%model/up_sampling3d_5/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splite
#model/up_sampling3d_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_5/concat_1ConcatV2&model/up_sampling3d_5/split_1:output:0&model/up_sampling3d_5/split_1:output:0&model/up_sampling3d_5/split_1:output:1&model/up_sampling3d_5/split_1:output:1&model/up_sampling3d_5/split_1:output:2&model/up_sampling3d_5/split_1:output:2&model/up_sampling3d_5/split_1:output:3&model/up_sampling3d_5/split_1:output:3&model/up_sampling3d_5/split_1:output:4&model/up_sampling3d_5/split_1:output:4&model/up_sampling3d_5/split_1:output:5&model/up_sampling3d_5/split_1:output:5&model/up_sampling3d_5/split_1:output:6&model/up_sampling3d_5/split_1:output:6&model/up_sampling3d_5/split_1:output:7&model/up_sampling3d_5/split_1:output:7&model/up_sampling3d_5/split_1:output:8&model/up_sampling3d_5/split_1:output:8&model/up_sampling3d_5/split_1:output:9&model/up_sampling3d_5/split_1:output:9'model/up_sampling3d_5/split_1:output:10'model/up_sampling3d_5/split_1:output:10'model/up_sampling3d_5/split_1:output:11'model/up_sampling3d_5/split_1:output:11'model/up_sampling3d_5/split_1:output:12'model/up_sampling3d_5/split_1:output:12'model/up_sampling3d_5/split_1:output:13'model/up_sampling3d_5/split_1:output:13'model/up_sampling3d_5/split_1:output:14'model/up_sampling3d_5/split_1:output:14'model/up_sampling3d_5/split_1:output:15'model/up_sampling3d_5/split_1:output:15,model/up_sampling3d_5/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  i
'model/up_sampling3d_5/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_5/split_2Split0model/up_sampling3d_5/split_2/split_dim:output:0'model/up_sampling3d_5/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splite
#model/up_sampling3d_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_5/concat_2ConcatV2&model/up_sampling3d_5/split_2:output:0&model/up_sampling3d_5/split_2:output:0&model/up_sampling3d_5/split_2:output:1&model/up_sampling3d_5/split_2:output:1&model/up_sampling3d_5/split_2:output:2&model/up_sampling3d_5/split_2:output:2&model/up_sampling3d_5/split_2:output:3&model/up_sampling3d_5/split_2:output:3&model/up_sampling3d_5/split_2:output:4&model/up_sampling3d_5/split_2:output:4&model/up_sampling3d_5/split_2:output:5&model/up_sampling3d_5/split_2:output:5&model/up_sampling3d_5/split_2:output:6&model/up_sampling3d_5/split_2:output:6&model/up_sampling3d_5/split_2:output:7&model/up_sampling3d_5/split_2:output:7&model/up_sampling3d_5/split_2:output:8&model/up_sampling3d_5/split_2:output:8&model/up_sampling3d_5/split_2:output:9&model/up_sampling3d_5/split_2:output:9'model/up_sampling3d_5/split_2:output:10'model/up_sampling3d_5/split_2:output:10'model/up_sampling3d_5/split_2:output:11'model/up_sampling3d_5/split_2:output:11'model/up_sampling3d_5/split_2:output:12'model/up_sampling3d_5/split_2:output:12'model/up_sampling3d_5/split_2:output:13'model/up_sampling3d_5/split_2:output:13'model/up_sampling3d_5/split_2:output:14'model/up_sampling3d_5/split_2:output:14'model/up_sampling3d_5/split_2:output:15'model/up_sampling3d_5/split_2:output:15,model/up_sampling3d_5/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   g
%model/up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_3/splitSplit.model/up_sampling3d_3/split/split_dim:output:0"model/conv3d_22/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�

model/up_sampling3d_3/concatConcatV2$model/up_sampling3d_3/split:output:0$model/up_sampling3d_3/split:output:0$model/up_sampling3d_3/split:output:1$model/up_sampling3d_3/split:output:1$model/up_sampling3d_3/split:output:2$model/up_sampling3d_3/split:output:2$model/up_sampling3d_3/split:output:3$model/up_sampling3d_3/split:output:3$model/up_sampling3d_3/split:output:4$model/up_sampling3d_3/split:output:4$model/up_sampling3d_3/split:output:5$model/up_sampling3d_3/split:output:5$model/up_sampling3d_3/split:output:6$model/up_sampling3d_3/split:output:6$model/up_sampling3d_3/split:output:7$model/up_sampling3d_3/split:output:7$model/up_sampling3d_3/split:output:8$model/up_sampling3d_3/split:output:8$model/up_sampling3d_3/split:output:9$model/up_sampling3d_3/split:output:9%model/up_sampling3d_3/split:output:10%model/up_sampling3d_3/split:output:10%model/up_sampling3d_3/split:output:11%model/up_sampling3d_3/split:output:11%model/up_sampling3d_3/split:output:12%model/up_sampling3d_3/split:output:12%model/up_sampling3d_3/split:output:13%model/up_sampling3d_3/split:output:13%model/up_sampling3d_3/split:output:14%model/up_sampling3d_3/split:output:14%model/up_sampling3d_3/split:output:15%model/up_sampling3d_3/split:output:15*model/up_sampling3d_3/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� i
'model/up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_3/split_1Split0model/up_sampling3d_3/split_1/split_dim:output:0%model/up_sampling3d_3/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splite
#model/up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_3/concat_1ConcatV2&model/up_sampling3d_3/split_1:output:0&model/up_sampling3d_3/split_1:output:0&model/up_sampling3d_3/split_1:output:1&model/up_sampling3d_3/split_1:output:1&model/up_sampling3d_3/split_1:output:2&model/up_sampling3d_3/split_1:output:2&model/up_sampling3d_3/split_1:output:3&model/up_sampling3d_3/split_1:output:3&model/up_sampling3d_3/split_1:output:4&model/up_sampling3d_3/split_1:output:4&model/up_sampling3d_3/split_1:output:5&model/up_sampling3d_3/split_1:output:5&model/up_sampling3d_3/split_1:output:6&model/up_sampling3d_3/split_1:output:6&model/up_sampling3d_3/split_1:output:7&model/up_sampling3d_3/split_1:output:7&model/up_sampling3d_3/split_1:output:8&model/up_sampling3d_3/split_1:output:8&model/up_sampling3d_3/split_1:output:9&model/up_sampling3d_3/split_1:output:9'model/up_sampling3d_3/split_1:output:10'model/up_sampling3d_3/split_1:output:10'model/up_sampling3d_3/split_1:output:11'model/up_sampling3d_3/split_1:output:11'model/up_sampling3d_3/split_1:output:12'model/up_sampling3d_3/split_1:output:12'model/up_sampling3d_3/split_1:output:13'model/up_sampling3d_3/split_1:output:13'model/up_sampling3d_3/split_1:output:14'model/up_sampling3d_3/split_1:output:14'model/up_sampling3d_3/split_1:output:15'model/up_sampling3d_3/split_1:output:15,model/up_sampling3d_3/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  i
'model/up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_3/split_2Split0model/up_sampling3d_3/split_2/split_dim:output:0'model/up_sampling3d_3/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splite
#model/up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_3/concat_2ConcatV2&model/up_sampling3d_3/split_2:output:0&model/up_sampling3d_3/split_2:output:0&model/up_sampling3d_3/split_2:output:1&model/up_sampling3d_3/split_2:output:1&model/up_sampling3d_3/split_2:output:2&model/up_sampling3d_3/split_2:output:2&model/up_sampling3d_3/split_2:output:3&model/up_sampling3d_3/split_2:output:3&model/up_sampling3d_3/split_2:output:4&model/up_sampling3d_3/split_2:output:4&model/up_sampling3d_3/split_2:output:5&model/up_sampling3d_3/split_2:output:5&model/up_sampling3d_3/split_2:output:6&model/up_sampling3d_3/split_2:output:6&model/up_sampling3d_3/split_2:output:7&model/up_sampling3d_3/split_2:output:7&model/up_sampling3d_3/split_2:output:8&model/up_sampling3d_3/split_2:output:8&model/up_sampling3d_3/split_2:output:9&model/up_sampling3d_3/split_2:output:9'model/up_sampling3d_3/split_2:output:10'model/up_sampling3d_3/split_2:output:10'model/up_sampling3d_3/split_2:output:11'model/up_sampling3d_3/split_2:output:11'model/up_sampling3d_3/split_2:output:12'model/up_sampling3d_3/split_2:output:12'model/up_sampling3d_3/split_2:output:13'model/up_sampling3d_3/split_2:output:13'model/up_sampling3d_3/split_2:output:14'model/up_sampling3d_3/split_2:output:14'model/up_sampling3d_3/split_2:output:15'model/up_sampling3d_3/split_2:output:15,model/up_sampling3d_3/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   g
%model/up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_1/splitSplit.model/up_sampling3d_1/split/split_dim:output:0"model/conv3d_17/Relu:activations:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitc
!model/up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�

model/up_sampling3d_1/concatConcatV2$model/up_sampling3d_1/split:output:0$model/up_sampling3d_1/split:output:0$model/up_sampling3d_1/split:output:1$model/up_sampling3d_1/split:output:1$model/up_sampling3d_1/split:output:2$model/up_sampling3d_1/split:output:2$model/up_sampling3d_1/split:output:3$model/up_sampling3d_1/split:output:3$model/up_sampling3d_1/split:output:4$model/up_sampling3d_1/split:output:4$model/up_sampling3d_1/split:output:5$model/up_sampling3d_1/split:output:5$model/up_sampling3d_1/split:output:6$model/up_sampling3d_1/split:output:6$model/up_sampling3d_1/split:output:7$model/up_sampling3d_1/split:output:7$model/up_sampling3d_1/split:output:8$model/up_sampling3d_1/split:output:8$model/up_sampling3d_1/split:output:9$model/up_sampling3d_1/split:output:9%model/up_sampling3d_1/split:output:10%model/up_sampling3d_1/split:output:10%model/up_sampling3d_1/split:output:11%model/up_sampling3d_1/split:output:11%model/up_sampling3d_1/split:output:12%model/up_sampling3d_1/split:output:12%model/up_sampling3d_1/split:output:13%model/up_sampling3d_1/split:output:13%model/up_sampling3d_1/split:output:14%model/up_sampling3d_1/split:output:14%model/up_sampling3d_1/split:output:15%model/up_sampling3d_1/split:output:15*model/up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� i
'model/up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_1/split_1Split0model/up_sampling3d_1/split_1/split_dim:output:0%model/up_sampling3d_1/concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splite
#model/up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_1/concat_1ConcatV2&model/up_sampling3d_1/split_1:output:0&model/up_sampling3d_1/split_1:output:0&model/up_sampling3d_1/split_1:output:1&model/up_sampling3d_1/split_1:output:1&model/up_sampling3d_1/split_1:output:2&model/up_sampling3d_1/split_1:output:2&model/up_sampling3d_1/split_1:output:3&model/up_sampling3d_1/split_1:output:3&model/up_sampling3d_1/split_1:output:4&model/up_sampling3d_1/split_1:output:4&model/up_sampling3d_1/split_1:output:5&model/up_sampling3d_1/split_1:output:5&model/up_sampling3d_1/split_1:output:6&model/up_sampling3d_1/split_1:output:6&model/up_sampling3d_1/split_1:output:7&model/up_sampling3d_1/split_1:output:7&model/up_sampling3d_1/split_1:output:8&model/up_sampling3d_1/split_1:output:8&model/up_sampling3d_1/split_1:output:9&model/up_sampling3d_1/split_1:output:9'model/up_sampling3d_1/split_1:output:10'model/up_sampling3d_1/split_1:output:10'model/up_sampling3d_1/split_1:output:11'model/up_sampling3d_1/split_1:output:11'model/up_sampling3d_1/split_1:output:12'model/up_sampling3d_1/split_1:output:12'model/up_sampling3d_1/split_1:output:13'model/up_sampling3d_1/split_1:output:13'model/up_sampling3d_1/split_1:output:14'model/up_sampling3d_1/split_1:output:14'model/up_sampling3d_1/split_1:output:15'model/up_sampling3d_1/split_1:output:15,model/up_sampling3d_1/concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  i
'model/up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_1/split_2Split0model/up_sampling3d_1/split_2/split_dim:output:0'model/up_sampling3d_1/concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splite
#model/up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/up_sampling3d_1/concat_2ConcatV2&model/up_sampling3d_1/split_2:output:0&model/up_sampling3d_1/split_2:output:0&model/up_sampling3d_1/split_2:output:1&model/up_sampling3d_1/split_2:output:1&model/up_sampling3d_1/split_2:output:2&model/up_sampling3d_1/split_2:output:2&model/up_sampling3d_1/split_2:output:3&model/up_sampling3d_1/split_2:output:3&model/up_sampling3d_1/split_2:output:4&model/up_sampling3d_1/split_2:output:4&model/up_sampling3d_1/split_2:output:5&model/up_sampling3d_1/split_2:output:5&model/up_sampling3d_1/split_2:output:6&model/up_sampling3d_1/split_2:output:6&model/up_sampling3d_1/split_2:output:7&model/up_sampling3d_1/split_2:output:7&model/up_sampling3d_1/split_2:output:8&model/up_sampling3d_1/split_2:output:8&model/up_sampling3d_1/split_2:output:9&model/up_sampling3d_1/split_2:output:9'model/up_sampling3d_1/split_2:output:10'model/up_sampling3d_1/split_2:output:10'model/up_sampling3d_1/split_2:output:11'model/up_sampling3d_1/split_2:output:11'model/up_sampling3d_1/split_2:output:12'model/up_sampling3d_1/split_2:output:12'model/up_sampling3d_1/split_2:output:13'model/up_sampling3d_1/split_2:output:13'model/up_sampling3d_1/split_2:output:14'model/up_sampling3d_1/split_2:output:14'model/up_sampling3d_1/split_2:output:15'model/up_sampling3d_1/split_2:output:15,model/up_sampling3d_1/concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   �
%model/conv3d_28/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_28_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_28/Conv3DConv3D'model/up_sampling3d_5/concat_2:output:0-model/conv3d_28/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_28/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_28/BiasAddBiasAddmodel/conv3d_28/Conv3D:output:0.model/conv3d_28/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_28/ReluRelu model/conv3d_28/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_23/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_23_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_23/Conv3DConv3D'model/up_sampling3d_3/concat_2:output:0-model/conv3d_23/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_23/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_23/BiasAddBiasAddmodel/conv3d_23/Conv3D:output:0.model/conv3d_23/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_23/ReluRelu model/conv3d_23/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_18/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_18/Conv3DConv3D'model/up_sampling3d_1/concat_2:output:0-model/conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_18/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_18/BiasAddBiasAddmodel/conv3d_18/Conv3D:output:0.model/conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_18/ReluRelu model/conv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_19/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_19/Conv3DConv3D"model/conv3d_18/Relu:activations:0-model/conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_19/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_19/BiasAddBiasAddmodel/conv3d_19/Conv3D:output:0.model/conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_19/ReluRelu model/conv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_24/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_24_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_24/Conv3DConv3D"model/conv3d_23/Relu:activations:0-model/conv3d_24/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_24/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_24/BiasAddBiasAddmodel/conv3d_24/Conv3D:output:0.model/conv3d_24/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_24/ReluRelu model/conv3d_24/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_29/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_29_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_29/Conv3DConv3D"model/conv3d_28/Relu:activations:0-model/conv3d_29/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_29/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_29/BiasAddBiasAddmodel/conv3d_29/Conv3D:output:0.model/conv3d_29/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_29/ReluRelu model/conv3d_29/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
model/add_1/addAddV2"model/conv3d_19/Relu:activations:0"model/conv3d_24/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
model/add_1/add_1AddV2model/add_1/add:z:0"model/conv3d_29/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_30/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_30_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_30/Conv3DConv3Dmodel/add_1/add_1:z:0-model/conv3d_30/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_30/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_30/BiasAddBiasAddmodel/conv3d_30/Conv3D:output:0.model/conv3d_30/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   {
IdentityIdentity model/conv3d_30/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   �
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp'^model/conv3d_10/BiasAdd/ReadVariableOp&^model/conv3d_10/Conv3D/ReadVariableOp'^model/conv3d_11/BiasAdd/ReadVariableOp&^model/conv3d_11/Conv3D/ReadVariableOp'^model/conv3d_12/BiasAdd/ReadVariableOp&^model/conv3d_12/Conv3D/ReadVariableOp'^model/conv3d_13/BiasAdd/ReadVariableOp&^model/conv3d_13/Conv3D/ReadVariableOp'^model/conv3d_14/BiasAdd/ReadVariableOp&^model/conv3d_14/Conv3D/ReadVariableOp'^model/conv3d_15/BiasAdd/ReadVariableOp&^model/conv3d_15/Conv3D/ReadVariableOp'^model/conv3d_16/BiasAdd/ReadVariableOp&^model/conv3d_16/Conv3D/ReadVariableOp'^model/conv3d_17/BiasAdd/ReadVariableOp&^model/conv3d_17/Conv3D/ReadVariableOp'^model/conv3d_18/BiasAdd/ReadVariableOp&^model/conv3d_18/Conv3D/ReadVariableOp'^model/conv3d_19/BiasAdd/ReadVariableOp&^model/conv3d_19/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp'^model/conv3d_20/BiasAdd/ReadVariableOp&^model/conv3d_20/Conv3D/ReadVariableOp'^model/conv3d_21/BiasAdd/ReadVariableOp&^model/conv3d_21/Conv3D/ReadVariableOp'^model/conv3d_22/BiasAdd/ReadVariableOp&^model/conv3d_22/Conv3D/ReadVariableOp'^model/conv3d_23/BiasAdd/ReadVariableOp&^model/conv3d_23/Conv3D/ReadVariableOp'^model/conv3d_24/BiasAdd/ReadVariableOp&^model/conv3d_24/Conv3D/ReadVariableOp'^model/conv3d_25/BiasAdd/ReadVariableOp&^model/conv3d_25/Conv3D/ReadVariableOp'^model/conv3d_26/BiasAdd/ReadVariableOp&^model/conv3d_26/Conv3D/ReadVariableOp'^model/conv3d_27/BiasAdd/ReadVariableOp&^model/conv3d_27/Conv3D/ReadVariableOp'^model/conv3d_28/BiasAdd/ReadVariableOp&^model/conv3d_28/Conv3D/ReadVariableOp'^model/conv3d_29/BiasAdd/ReadVariableOp&^model/conv3d_29/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp'^model/conv3d_30/BiasAdd/ReadVariableOp&^model/conv3d_30/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp&^model/conv3d_8/BiasAdd/ReadVariableOp%^model/conv3d_8/Conv3D/ReadVariableOp&^model/conv3d_9/BiasAdd/ReadVariableOp%^model/conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
$model/conv3d_9/Conv3D/ReadVariableOp$model/conv3d_9/Conv3D/ReadVariableOp:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
)__inference_conv3d_12_layer_call_fn_95333

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_91439{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_19_layer_call_and_return_conditional_losses_96160

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_93937

inputs%
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
identity��StatefulPartitionedCall�	
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
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_92228{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_25_layer_call_and_return_conditional_losses_91651

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_20_layer_call_and_return_conditional_losses_91668

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_18_layer_call_and_return_conditional_losses_92144

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�%
f
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_96080

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_22_layer_call_and_return_conditional_losses_95862

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_16_layer_call_and_return_conditional_losses_91853

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_15_layer_call_and_return_conditional_losses_91685

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_20_layer_call_and_return_conditional_losses_95616

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91420

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_91473

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_22_layer_call_fn_95851

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_22_layer_call_and_return_conditional_losses_91887{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_5_layer_call_fn_95133

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_91336{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
܊
��
!__inference__traced_restore_97425
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
identity_194��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_154�AssignVariableOp_155�AssignVariableOp_156�AssignVariableOp_157�AssignVariableOp_158�AssignVariableOp_159�AssignVariableOp_16�AssignVariableOp_160�AssignVariableOp_161�AssignVariableOp_162�AssignVariableOp_163�AssignVariableOp_164�AssignVariableOp_165�AssignVariableOp_166�AssignVariableOp_167�AssignVariableOp_168�AssignVariableOp_169�AssignVariableOp_17�AssignVariableOp_170�AssignVariableOp_171�AssignVariableOp_172�AssignVariableOp_173�AssignVariableOp_174�AssignVariableOp_175�AssignVariableOp_176�AssignVariableOp_177�AssignVariableOp_178�AssignVariableOp_179�AssignVariableOp_18�AssignVariableOp_180�AssignVariableOp_181�AssignVariableOp_182�AssignVariableOp_183�AssignVariableOp_184�AssignVariableOp_185�AssignVariableOp_186�AssignVariableOp_187�AssignVariableOp_188�AssignVariableOp_189�AssignVariableOp_19�AssignVariableOp_190�AssignVariableOp_191�AssignVariableOp_192�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�o
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�o
value�nB�n�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv3d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv3d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv3d_11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv3d_11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv3d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv3d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv3d_12_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv3d_12_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv3d_8_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv3d_8_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv3d_13_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv3d_13_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv3d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv3d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv3d_9_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp!assignvariableop_27_conv3d_9_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv3d_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv3d_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv3d_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv3d_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv3d_20_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv3d_20_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv3d_25_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv3d_25_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv3d_16_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv3d_16_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv3d_21_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv3d_21_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv3d_26_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv3d_26_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv3d_17_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv3d_17_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp$assignvariableop_44_conv3d_22_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv3d_22_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp$assignvariableop_46_conv3d_27_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp"assignvariableop_47_conv3d_27_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv3d_18_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv3d_18_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp$assignvariableop_50_conv3d_23_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp"assignvariableop_51_conv3d_23_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp$assignvariableop_52_conv3d_28_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp"assignvariableop_53_conv3d_28_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp$assignvariableop_54_conv3d_19_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp"assignvariableop_55_conv3d_19_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp$assignvariableop_56_conv3d_24_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp"assignvariableop_57_conv3d_24_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp$assignvariableop_58_conv3d_29_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp"assignvariableop_59_conv3d_29_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp$assignvariableop_60_conv3d_30_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp"assignvariableop_61_conv3d_30_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_adam_iterIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_adam_beta_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpassignvariableop_64_adam_beta_2Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_decayIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp&assignvariableop_66_adam_learning_rateIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpassignvariableop_67_totalIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_countIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_conv3d_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_conv3d_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv3d_5_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv3d_5_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv3d_10_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv3d_10_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv3d_1_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv3d_1_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv3d_6_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv3d_6_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv3d_11_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv3d_11_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv3d_2_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv3d_2_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv3d_7_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv3d_7_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv3d_12_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv3d_12_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv3d_3_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv3d_3_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv3d_8_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv3d_8_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv3d_13_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv3d_13_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_conv3d_4_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_conv3d_4_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_conv3d_9_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_conv3d_9_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv3d_14_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv3d_14_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_conv3d_15_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_conv3d_15_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv3d_20_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv3d_20_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv3d_25_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv3d_25_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv3d_16_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv3d_16_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv3d_21_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv3d_21_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv3d_26_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv3d_26_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_conv3d_17_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_conv3d_17_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv3d_22_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv3d_22_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_conv3d_27_kernel_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_conv3d_27_bias_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv3d_18_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv3d_18_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_conv3d_23_kernel_mIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_conv3d_23_bias_mIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv3d_28_kernel_mIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv3d_28_bias_mIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_conv3d_19_kernel_mIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_conv3d_19_bias_mIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_conv3d_24_kernel_mIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_conv3d_24_bias_mIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_conv3d_29_kernel_mIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_conv3d_29_bias_mIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp,assignvariableop_129_adam_conv3d_30_kernel_mIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp*assignvariableop_130_adam_conv3d_30_bias_mIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp)assignvariableop_131_adam_conv3d_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp'assignvariableop_132_adam_conv3d_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp+assignvariableop_133_adam_conv3d_5_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOp)assignvariableop_134_adam_conv3d_5_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_conv3d_10_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_conv3d_10_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOp+assignvariableop_137_adam_conv3d_1_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOp)assignvariableop_138_adam_conv3d_1_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOp+assignvariableop_139_adam_conv3d_6_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOp)assignvariableop_140_adam_conv3d_6_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_141AssignVariableOp,assignvariableop_141_adam_conv3d_11_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_142AssignVariableOp*assignvariableop_142_adam_conv3d_11_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_143AssignVariableOp+assignvariableop_143_adam_conv3d_2_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_144AssignVariableOp)assignvariableop_144_adam_conv3d_2_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_145AssignVariableOp+assignvariableop_145_adam_conv3d_7_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_146AssignVariableOp)assignvariableop_146_adam_conv3d_7_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_147AssignVariableOp,assignvariableop_147_adam_conv3d_12_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_148AssignVariableOp*assignvariableop_148_adam_conv3d_12_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_149AssignVariableOp+assignvariableop_149_adam_conv3d_3_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_150AssignVariableOp)assignvariableop_150_adam_conv3d_3_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_151AssignVariableOp+assignvariableop_151_adam_conv3d_8_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_152AssignVariableOp)assignvariableop_152_adam_conv3d_8_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_153AssignVariableOp,assignvariableop_153_adam_conv3d_13_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_154AssignVariableOp*assignvariableop_154_adam_conv3d_13_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_155AssignVariableOp+assignvariableop_155_adam_conv3d_4_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_156AssignVariableOp)assignvariableop_156_adam_conv3d_4_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_157AssignVariableOp+assignvariableop_157_adam_conv3d_9_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_158AssignVariableOp)assignvariableop_158_adam_conv3d_9_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_159AssignVariableOp,assignvariableop_159_adam_conv3d_14_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_160AssignVariableOp*assignvariableop_160_adam_conv3d_14_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_161AssignVariableOp,assignvariableop_161_adam_conv3d_15_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_162AssignVariableOp*assignvariableop_162_adam_conv3d_15_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_163AssignVariableOp,assignvariableop_163_adam_conv3d_20_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_164AssignVariableOp*assignvariableop_164_adam_conv3d_20_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_165AssignVariableOp,assignvariableop_165_adam_conv3d_25_kernel_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_166AssignVariableOp*assignvariableop_166_adam_conv3d_25_bias_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_167AssignVariableOp,assignvariableop_167_adam_conv3d_16_kernel_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_168AssignVariableOp*assignvariableop_168_adam_conv3d_16_bias_vIdentity_168:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_169AssignVariableOp,assignvariableop_169_adam_conv3d_21_kernel_vIdentity_169:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_170AssignVariableOp*assignvariableop_170_adam_conv3d_21_bias_vIdentity_170:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_171IdentityRestoreV2:tensors:171"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_171AssignVariableOp,assignvariableop_171_adam_conv3d_26_kernel_vIdentity_171:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_172IdentityRestoreV2:tensors:172"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_172AssignVariableOp*assignvariableop_172_adam_conv3d_26_bias_vIdentity_172:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_173IdentityRestoreV2:tensors:173"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_173AssignVariableOp,assignvariableop_173_adam_conv3d_17_kernel_vIdentity_173:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_174IdentityRestoreV2:tensors:174"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_174AssignVariableOp*assignvariableop_174_adam_conv3d_17_bias_vIdentity_174:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_175IdentityRestoreV2:tensors:175"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_175AssignVariableOp,assignvariableop_175_adam_conv3d_22_kernel_vIdentity_175:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_176IdentityRestoreV2:tensors:176"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_176AssignVariableOp*assignvariableop_176_adam_conv3d_22_bias_vIdentity_176:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_177IdentityRestoreV2:tensors:177"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_177AssignVariableOp,assignvariableop_177_adam_conv3d_27_kernel_vIdentity_177:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_178IdentityRestoreV2:tensors:178"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_178AssignVariableOp*assignvariableop_178_adam_conv3d_27_bias_vIdentity_178:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_179IdentityRestoreV2:tensors:179"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_179AssignVariableOp,assignvariableop_179_adam_conv3d_18_kernel_vIdentity_179:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_180IdentityRestoreV2:tensors:180"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_180AssignVariableOp*assignvariableop_180_adam_conv3d_18_bias_vIdentity_180:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_181IdentityRestoreV2:tensors:181"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_181AssignVariableOp,assignvariableop_181_adam_conv3d_23_kernel_vIdentity_181:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_182IdentityRestoreV2:tensors:182"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_182AssignVariableOp*assignvariableop_182_adam_conv3d_23_bias_vIdentity_182:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_183IdentityRestoreV2:tensors:183"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_183AssignVariableOp,assignvariableop_183_adam_conv3d_28_kernel_vIdentity_183:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_184IdentityRestoreV2:tensors:184"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_184AssignVariableOp*assignvariableop_184_adam_conv3d_28_bias_vIdentity_184:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_185IdentityRestoreV2:tensors:185"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_185AssignVariableOp,assignvariableop_185_adam_conv3d_19_kernel_vIdentity_185:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_186IdentityRestoreV2:tensors:186"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_186AssignVariableOp*assignvariableop_186_adam_conv3d_19_bias_vIdentity_186:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_187IdentityRestoreV2:tensors:187"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_187AssignVariableOp,assignvariableop_187_adam_conv3d_24_kernel_vIdentity_187:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_188IdentityRestoreV2:tensors:188"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_188AssignVariableOp*assignvariableop_188_adam_conv3d_24_bias_vIdentity_188:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_189IdentityRestoreV2:tensors:189"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_189AssignVariableOp,assignvariableop_189_adam_conv3d_29_kernel_vIdentity_189:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_190IdentityRestoreV2:tensors:190"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_190AssignVariableOp*assignvariableop_190_adam_conv3d_29_bias_vIdentity_190:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_191IdentityRestoreV2:tensors:191"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_191AssignVariableOp,assignvariableop_191_adam_conv3d_30_kernel_vIdentity_191:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_192IdentityRestoreV2:tensors:192"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_192AssignVariableOp*assignvariableop_192_adam_conv3d_30_bias_vIdentity_192:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �"
Identity_193Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_186^AssignVariableOp_187^AssignVariableOp_188^AssignVariableOp_189^AssignVariableOp_19^AssignVariableOp_190^AssignVariableOp_191^AssignVariableOp_192^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_194IdentityIdentity_193:output:0^NoOp_1*
T0*
_output_shapes
: �"
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_186^AssignVariableOp_187^AssignVariableOp_188^AssignVariableOp_189^AssignVariableOp_19^AssignVariableOp_190^AssignVariableOp_191^AssignVariableOp_192^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_194Identity_194:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
�
x
@__inference_add_1_layer_call_and_return_conditional_losses_92209

inputs
inputs_1
inputs_2
identity\
addAddV2inputsinputs_1*
T0*3
_output_shapes!
:���������   _
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:���������   ]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:���������   :���������   :���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
C
'__inference_reshape_layer_call_fn_95544

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_91621a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91262

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_18_layer_call_fn_96089

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_92144{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
d
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_95678

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_95259

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_11_layer_call_and_return_conditional_losses_95224

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_95204

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_12_layer_call_and_return_conditional_losses_95344

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_23_layer_call_fn_96109

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_23_layer_call_and_return_conditional_losses_92127{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
A__inference_conv3d_layer_call_and_return_conditional_losses_91353

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
(__inference_conv3d_2_layer_call_fn_95293

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_91473{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_12_layer_call_and_return_conditional_losses_91439

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
]
#__inference_add_layer_call_fn_95531
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_91607l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:���������:���������:���������:] Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs/2
�
�
)__inference_conv3d_13_layer_call_fn_95393

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_91490{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_up_sampling3d_3_layer_call_fn_95953

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_92034l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_16_layer_call_fn_95771

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_91853{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_95239

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_93321
input_1%
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
identity��StatefulPartitionedCall�	
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
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_93065{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
)__inference_conv3d_27_layer_call_fn_95871

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_27_layer_call_and_return_conditional_losses_91870{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_95419

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�%
f
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_96014

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:��������� S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� :��������� *
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15concat_1/axis:output:0*
N *
T0*3
_output_shapes!
:���������  S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  :���������  *
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15concat_2/axis:output:0*
N *
T0*3
_output_shapes!
:���������   e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_3_layer_call_fn_95434

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91540l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
x
>__inference_add_layer_call_and_return_conditional_losses_95539
inputs_0
inputs_1
inputs_2
identity^
addAddV2inputs_0inputs_1*
T0*3
_output_shapes!
:���������_
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:���������]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:���������:���������:���������:] Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs/1:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs/2
�
�
)__inference_conv3d_29_layer_call_fn_96189

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_29_layer_call_and_return_conditional_losses_92195{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
(__inference_conv3d_3_layer_call_fn_95353

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_91524{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_22_layer_call_and_return_conditional_losses_91887

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_91336

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
��
�O
__inference__traced_save_96836
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

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �o
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�o
value�nB�n�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-30/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-30/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �K
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop+savev2_conv3d_10_kernel_read_readvariableop)savev2_conv3d_10_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop+savev2_conv3d_11_kernel_read_readvariableop)savev2_conv3d_11_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop+savev2_conv3d_12_kernel_read_readvariableop)savev2_conv3d_12_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_8_kernel_read_readvariableop(savev2_conv3d_8_bias_read_readvariableop+savev2_conv3d_13_kernel_read_readvariableop)savev2_conv3d_13_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_9_kernel_read_readvariableop(savev2_conv3d_9_bias_read_readvariableop+savev2_conv3d_14_kernel_read_readvariableop)savev2_conv3d_14_bias_read_readvariableop+savev2_conv3d_15_kernel_read_readvariableop)savev2_conv3d_15_bias_read_readvariableop+savev2_conv3d_20_kernel_read_readvariableop)savev2_conv3d_20_bias_read_readvariableop+savev2_conv3d_25_kernel_read_readvariableop)savev2_conv3d_25_bias_read_readvariableop+savev2_conv3d_16_kernel_read_readvariableop)savev2_conv3d_16_bias_read_readvariableop+savev2_conv3d_21_kernel_read_readvariableop)savev2_conv3d_21_bias_read_readvariableop+savev2_conv3d_26_kernel_read_readvariableop)savev2_conv3d_26_bias_read_readvariableop+savev2_conv3d_17_kernel_read_readvariableop)savev2_conv3d_17_bias_read_readvariableop+savev2_conv3d_22_kernel_read_readvariableop)savev2_conv3d_22_bias_read_readvariableop+savev2_conv3d_27_kernel_read_readvariableop)savev2_conv3d_27_bias_read_readvariableop+savev2_conv3d_18_kernel_read_readvariableop)savev2_conv3d_18_bias_read_readvariableop+savev2_conv3d_23_kernel_read_readvariableop)savev2_conv3d_23_bias_read_readvariableop+savev2_conv3d_28_kernel_read_readvariableop)savev2_conv3d_28_bias_read_readvariableop+savev2_conv3d_19_kernel_read_readvariableop)savev2_conv3d_19_bias_read_readvariableop+savev2_conv3d_24_kernel_read_readvariableop)savev2_conv3d_24_bias_read_readvariableop+savev2_conv3d_29_kernel_read_readvariableop)savev2_conv3d_29_bias_read_readvariableop+savev2_conv3d_30_kernel_read_readvariableop)savev2_conv3d_30_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop2savev2_adam_conv3d_10_kernel_m_read_readvariableop0savev2_adam_conv3d_10_bias_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop1savev2_adam_conv3d_6_kernel_m_read_readvariableop/savev2_adam_conv3d_6_bias_m_read_readvariableop2savev2_adam_conv3d_11_kernel_m_read_readvariableop0savev2_adam_conv3d_11_bias_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop1savev2_adam_conv3d_7_kernel_m_read_readvariableop/savev2_adam_conv3d_7_bias_m_read_readvariableop2savev2_adam_conv3d_12_kernel_m_read_readvariableop0savev2_adam_conv3d_12_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_8_kernel_m_read_readvariableop/savev2_adam_conv3d_8_bias_m_read_readvariableop2savev2_adam_conv3d_13_kernel_m_read_readvariableop0savev2_adam_conv3d_13_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_conv3d_9_kernel_m_read_readvariableop/savev2_adam_conv3d_9_bias_m_read_readvariableop2savev2_adam_conv3d_14_kernel_m_read_readvariableop0savev2_adam_conv3d_14_bias_m_read_readvariableop2savev2_adam_conv3d_15_kernel_m_read_readvariableop0savev2_adam_conv3d_15_bias_m_read_readvariableop2savev2_adam_conv3d_20_kernel_m_read_readvariableop0savev2_adam_conv3d_20_bias_m_read_readvariableop2savev2_adam_conv3d_25_kernel_m_read_readvariableop0savev2_adam_conv3d_25_bias_m_read_readvariableop2savev2_adam_conv3d_16_kernel_m_read_readvariableop0savev2_adam_conv3d_16_bias_m_read_readvariableop2savev2_adam_conv3d_21_kernel_m_read_readvariableop0savev2_adam_conv3d_21_bias_m_read_readvariableop2savev2_adam_conv3d_26_kernel_m_read_readvariableop0savev2_adam_conv3d_26_bias_m_read_readvariableop2savev2_adam_conv3d_17_kernel_m_read_readvariableop0savev2_adam_conv3d_17_bias_m_read_readvariableop2savev2_adam_conv3d_22_kernel_m_read_readvariableop0savev2_adam_conv3d_22_bias_m_read_readvariableop2savev2_adam_conv3d_27_kernel_m_read_readvariableop0savev2_adam_conv3d_27_bias_m_read_readvariableop2savev2_adam_conv3d_18_kernel_m_read_readvariableop0savev2_adam_conv3d_18_bias_m_read_readvariableop2savev2_adam_conv3d_23_kernel_m_read_readvariableop0savev2_adam_conv3d_23_bias_m_read_readvariableop2savev2_adam_conv3d_28_kernel_m_read_readvariableop0savev2_adam_conv3d_28_bias_m_read_readvariableop2savev2_adam_conv3d_19_kernel_m_read_readvariableop0savev2_adam_conv3d_19_bias_m_read_readvariableop2savev2_adam_conv3d_24_kernel_m_read_readvariableop0savev2_adam_conv3d_24_bias_m_read_readvariableop2savev2_adam_conv3d_29_kernel_m_read_readvariableop0savev2_adam_conv3d_29_bias_m_read_readvariableop2savev2_adam_conv3d_30_kernel_m_read_readvariableop0savev2_adam_conv3d_30_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop2savev2_adam_conv3d_10_kernel_v_read_readvariableop0savev2_adam_conv3d_10_bias_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop1savev2_adam_conv3d_6_kernel_v_read_readvariableop/savev2_adam_conv3d_6_bias_v_read_readvariableop2savev2_adam_conv3d_11_kernel_v_read_readvariableop0savev2_adam_conv3d_11_bias_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop1savev2_adam_conv3d_7_kernel_v_read_readvariableop/savev2_adam_conv3d_7_bias_v_read_readvariableop2savev2_adam_conv3d_12_kernel_v_read_readvariableop0savev2_adam_conv3d_12_bias_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_8_kernel_v_read_readvariableop/savev2_adam_conv3d_8_bias_v_read_readvariableop2savev2_adam_conv3d_13_kernel_v_read_readvariableop0savev2_adam_conv3d_13_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_conv3d_9_kernel_v_read_readvariableop/savev2_adam_conv3d_9_bias_v_read_readvariableop2savev2_adam_conv3d_14_kernel_v_read_readvariableop0savev2_adam_conv3d_14_bias_v_read_readvariableop2savev2_adam_conv3d_15_kernel_v_read_readvariableop0savev2_adam_conv3d_15_bias_v_read_readvariableop2savev2_adam_conv3d_20_kernel_v_read_readvariableop0savev2_adam_conv3d_20_bias_v_read_readvariableop2savev2_adam_conv3d_25_kernel_v_read_readvariableop0savev2_adam_conv3d_25_bias_v_read_readvariableop2savev2_adam_conv3d_16_kernel_v_read_readvariableop0savev2_adam_conv3d_16_bias_v_read_readvariableop2savev2_adam_conv3d_21_kernel_v_read_readvariableop0savev2_adam_conv3d_21_bias_v_read_readvariableop2savev2_adam_conv3d_26_kernel_v_read_readvariableop0savev2_adam_conv3d_26_bias_v_read_readvariableop2savev2_adam_conv3d_17_kernel_v_read_readvariableop0savev2_adam_conv3d_17_bias_v_read_readvariableop2savev2_adam_conv3d_22_kernel_v_read_readvariableop0savev2_adam_conv3d_22_bias_v_read_readvariableop2savev2_adam_conv3d_27_kernel_v_read_readvariableop0savev2_adam_conv3d_27_bias_v_read_readvariableop2savev2_adam_conv3d_18_kernel_v_read_readvariableop0savev2_adam_conv3d_18_bias_v_read_readvariableop2savev2_adam_conv3d_23_kernel_v_read_readvariableop0savev2_adam_conv3d_23_bias_v_read_readvariableop2savev2_adam_conv3d_28_kernel_v_read_readvariableop0savev2_adam_conv3d_28_bias_v_read_readvariableop2savev2_adam_conv3d_19_kernel_v_read_readvariableop0savev2_adam_conv3d_19_bias_v_read_readvariableop2savev2_adam_conv3d_24_kernel_v_read_readvariableop0savev2_adam_conv3d_24_bias_v_read_readvariableop2savev2_adam_conv3d_29_kernel_v_read_readvariableop0savev2_adam_conv3d_29_bias_v_read_readvariableop2savev2_adam_conv3d_30_kernel_v_read_readvariableop0savev2_adam_conv3d_30_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: : : : : : : ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 2(
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
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::1�,
*
_output_shapes
::!�

_output_shapes
::�

_output_shapes
: 
�
f
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_91767

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_92355
input_1%
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
identity��StatefulPartitionedCall�	
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
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_92228{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
K
/__inference_max_pooling3d_2_layer_call_fn_95249

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91250�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_95459

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_9_layer_call_and_return_conditional_losses_91576

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:���������c
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_1_layer_call_fn_95409

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91274�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_91387

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
f
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_91728

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_93808
input_1%
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
identity��StatefulPartitionedCall�	
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
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_91229{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
��
�
@__inference_model_layer_call_and_return_conditional_losses_93496
input_1-
conv3d_10_93324:
conv3d_10_93326:,
conv3d_5_93329:
conv3d_5_93331:*
conv3d_93334:
conv3d_93336:-
conv3d_11_93339:
conv3d_11_93341:,
conv3d_6_93344:
conv3d_6_93346:,
conv3d_1_93349:
conv3d_1_93351:-
conv3d_12_93357:
conv3d_12_93359:,
conv3d_7_93362:
conv3d_7_93364:,
conv3d_2_93367:
conv3d_2_93369:-
conv3d_13_93372:
conv3d_13_93374:,
conv3d_8_93377:
conv3d_8_93379:,
conv3d_3_93382:
conv3d_3_93384:,
conv3d_4_93390:
conv3d_4_93392:,
conv3d_9_93395:
conv3d_9_93397:-
conv3d_14_93400:
conv3d_14_93402:-
conv3d_25_93408:
conv3d_25_93410:-
conv3d_20_93413:
conv3d_20_93415:-
conv3d_15_93418:
conv3d_15_93420:-
conv3d_26_93426:
conv3d_26_93428:-
conv3d_21_93431:
conv3d_21_93433:-
conv3d_16_93436:
conv3d_16_93438:-
conv3d_27_93441:
conv3d_27_93443:-
conv3d_22_93446:
conv3d_22_93448:-
conv3d_17_93451:
conv3d_17_93453:-
conv3d_28_93459:
conv3d_28_93461:-
conv3d_23_93464:
conv3d_23_93466:-
conv3d_18_93469:
conv3d_18_93471:-
conv3d_19_93474:
conv3d_19_93476:-
conv3d_24_93479:
conv3d_24_93481:-
conv3d_29_93484:
conv3d_29_93486:-
conv3d_30_93490:
conv3d_30_93492:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall�!conv3d_18/StatefulPartitionedCall�!conv3d_19/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall�!conv3d_20/StatefulPartitionedCall�!conv3d_21/StatefulPartitionedCall�!conv3d_22/StatefulPartitionedCall�!conv3d_23/StatefulPartitionedCall�!conv3d_24/StatefulPartitionedCall�!conv3d_25/StatefulPartitionedCall�!conv3d_26/StatefulPartitionedCall�!conv3d_27/StatefulPartitionedCall�!conv3d_28/StatefulPartitionedCall�!conv3d_29/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�!conv3d_30/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_10_93324conv3d_10_93326*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_91319�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_5_93329conv3d_5_93331*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_91336�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_93334conv3d_93336*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_91353�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_93339conv3d_11_93341*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_91370�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_93344conv3d_6_93346*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_91387�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_93349conv3d_1_93351*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_91404�
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91414�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91420�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91426�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_93357conv3d_12_93359*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_91439�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_93362conv3d_7_93364*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_91456�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_93367conv3d_2_93369*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_91473�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_93372conv3d_13_93374*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_91490�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_93377conv3d_8_93379*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_91507�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_93382conv3d_3_93384*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_91524�
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91534�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91540�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91546�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_93390conv3d_4_93392*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_91559�
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_93395conv3d_9_93397*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_91576�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_93400conv3d_14_93402*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_91593�
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_91607�
reshape/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_91621�
reshape_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_91638�
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_25_93408conv3d_25_93410*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_25_layer_call_and_return_conditional_losses_91651�
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_20_93413conv3d_20_93415*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_20_layer_call_and_return_conditional_losses_91668�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_15_93418conv3d_15_93420*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_91685�
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_91728�
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_91767�
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_91806�
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_93426conv3d_26_93428*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_26_layer_call_and_return_conditional_losses_91819�
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_93431conv3d_21_93433*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_21_layer_call_and_return_conditional_losses_91836�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_93436conv3d_16_93438*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_91853�
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_93441conv3d_27_93443*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_27_layer_call_and_return_conditional_losses_91870�
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_93446conv3d_22_93448*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_22_layer_call_and_return_conditional_losses_91887�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_93451conv3d_17_93453*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_91904�
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_91971�
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_92034�
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_92097�
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_93459conv3d_28_93461*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_28_layer_call_and_return_conditional_losses_92110�
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_93464conv3d_23_93466*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_23_layer_call_and_return_conditional_losses_92127�
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_93469conv3d_18_93471*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_92144�
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_93474conv3d_19_93476*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_19_layer_call_and_return_conditional_losses_92161�
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_93479conv3d_24_93481*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_24_layer_call_and_return_conditional_losses_92178�
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_93484conv3d_29_93486*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_29_layer_call_and_return_conditional_losses_92195�
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_92209�
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_93490conv3d_30_93492*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_30_layer_call_and_return_conditional_losses_92221�
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   �	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
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
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_91638

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
valueB:�
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
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:p
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������d
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_8_layer_call_and_return_conditional_losses_95384

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_27_layer_call_and_return_conditional_losses_91870

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_7_layer_call_fn_95313

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_91456{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
v
>__inference_add_layer_call_and_return_conditional_losses_91607

inputs
inputs_1
inputs_2
identity\
addAddV2inputsinputs_1*
T0*3
_output_shapes!
:���������_
add_1AddV2add:z:0inputs_2*
T0*3
_output_shapes!
:���������]
IdentityIdentity	add_1:z:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:���������:���������:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_95439

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_95762

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0inputs*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:���������S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:���������e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_16_layer_call_and_return_conditional_losses_95782

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_10_layer_call_and_return_conditional_losses_91319

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
K
/__inference_up_sampling3d_4_layer_call_fn_95725

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_91728l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_95424

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_13_layer_call_and_return_conditional_losses_95404

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_4_layer_call_fn_95269

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91262�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91534

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_1_layer_call_fn_95173

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_91404{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_19_layer_call_and_return_conditional_losses_92161

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_95184

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
K
/__inference_up_sampling3d_2_layer_call_fn_95683

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_91767l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_91456

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_4_layer_call_fn_95274

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91414l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_25_layer_call_and_return_conditional_losses_95636

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_28_layer_call_and_return_conditional_losses_96140

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_95324

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_8_layer_call_fn_95373

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_91507{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_92228

inputs-
conv3d_10_91320:
conv3d_10_91322:,
conv3d_5_91337:
conv3d_5_91339:*
conv3d_91354:
conv3d_91356:-
conv3d_11_91371:
conv3d_11_91373:,
conv3d_6_91388:
conv3d_6_91390:,
conv3d_1_91405:
conv3d_1_91407:-
conv3d_12_91440:
conv3d_12_91442:,
conv3d_7_91457:
conv3d_7_91459:,
conv3d_2_91474:
conv3d_2_91476:-
conv3d_13_91491:
conv3d_13_91493:,
conv3d_8_91508:
conv3d_8_91510:,
conv3d_3_91525:
conv3d_3_91527:,
conv3d_4_91560:
conv3d_4_91562:,
conv3d_9_91577:
conv3d_9_91579:-
conv3d_14_91594:
conv3d_14_91596:-
conv3d_25_91652:
conv3d_25_91654:-
conv3d_20_91669:
conv3d_20_91671:-
conv3d_15_91686:
conv3d_15_91688:-
conv3d_26_91820:
conv3d_26_91822:-
conv3d_21_91837:
conv3d_21_91839:-
conv3d_16_91854:
conv3d_16_91856:-
conv3d_27_91871:
conv3d_27_91873:-
conv3d_22_91888:
conv3d_22_91890:-
conv3d_17_91905:
conv3d_17_91907:-
conv3d_28_92111:
conv3d_28_92113:-
conv3d_23_92128:
conv3d_23_92130:-
conv3d_18_92145:
conv3d_18_92147:-
conv3d_19_92162:
conv3d_19_92164:-
conv3d_24_92179:
conv3d_24_92181:-
conv3d_29_92196:
conv3d_29_92198:-
conv3d_30_92222:
conv3d_30_92224:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall�!conv3d_18/StatefulPartitionedCall�!conv3d_19/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall�!conv3d_20/StatefulPartitionedCall�!conv3d_21/StatefulPartitionedCall�!conv3d_22/StatefulPartitionedCall�!conv3d_23/StatefulPartitionedCall�!conv3d_24/StatefulPartitionedCall�!conv3d_25/StatefulPartitionedCall�!conv3d_26/StatefulPartitionedCall�!conv3d_27/StatefulPartitionedCall�!conv3d_28/StatefulPartitionedCall�!conv3d_29/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�!conv3d_30/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_10_91320conv3d_10_91322*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_91319�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_5_91337conv3d_5_91339*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_91336�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_91354conv3d_91356*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_91353�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_91371conv3d_11_91373*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_91370�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_91388conv3d_6_91390*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_91387�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_91405conv3d_1_91407*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_91404�
max_pooling3d_4/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_91414�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91420�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_91426�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_4/PartitionedCall:output:0conv3d_12_91440conv3d_12_91442*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_91439�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_7_91457conv3d_7_91459*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_91456�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_91474conv3d_2_91476*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_91473�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_91491conv3d_13_91493*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_91490�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_91508conv3d_8_91510*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_91507�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_91525conv3d_3_91527*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_91524�
max_pooling3d_5/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_91534�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91540�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91546�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_91560conv3d_4_91562*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_91559�
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0conv3d_9_91577conv3d_9_91579*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_91576�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0conv3d_14_91594conv3d_14_91596*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_91593�
add/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0)conv3d_9/StatefulPartitionedCall:output:0*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_91607�
reshape/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_91621�
reshape_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_91638�
!conv3d_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_25_91652conv3d_25_91654*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_25_layer_call_and_return_conditional_losses_91651�
!conv3d_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_20_91669conv3d_20_91671*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_20_layer_call_and_return_conditional_losses_91668�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv3d_15_91686conv3d_15_91688*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_91685�
up_sampling3d_4/PartitionedCallPartitionedCall*conv3d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_91728�
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_91767�
up_sampling3d/PartitionedCallPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_91806�
!conv3d_26/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_4/PartitionedCall:output:0conv3d_26_91820conv3d_26_91822*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_26_layer_call_and_return_conditional_losses_91819�
!conv3d_21/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_21_91837conv3d_21_91839*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_21_layer_call_and_return_conditional_losses_91836�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_16_91854conv3d_16_91856*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_91853�
!conv3d_27/StatefulPartitionedCallStatefulPartitionedCall*conv3d_26/StatefulPartitionedCall:output:0conv3d_27_91871conv3d_27_91873*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_27_layer_call_and_return_conditional_losses_91870�
!conv3d_22/StatefulPartitionedCallStatefulPartitionedCall*conv3d_21/StatefulPartitionedCall:output:0conv3d_22_91888conv3d_22_91890*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_22_layer_call_and_return_conditional_losses_91887�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_91905conv3d_17_91907*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_91904�
up_sampling3d_5/PartitionedCallPartitionedCall*conv3d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_91971�
up_sampling3d_3/PartitionedCallPartitionedCall*conv3d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_92034�
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_92097�
!conv3d_28/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_5/PartitionedCall:output:0conv3d_28_92111conv3d_28_92113*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_28_layer_call_and_return_conditional_losses_92110�
!conv3d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_3/PartitionedCall:output:0conv3d_23_92128conv3d_23_92130*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_23_layer_call_and_return_conditional_losses_92127�
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_18_92145conv3d_18_92147*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_92144�
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0conv3d_19_92162conv3d_19_92164*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_19_layer_call_and_return_conditional_losses_92161�
!conv3d_24/StatefulPartitionedCallStatefulPartitionedCall*conv3d_23/StatefulPartitionedCall:output:0conv3d_24_92179conv3d_24_92181*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_24_layer_call_and_return_conditional_losses_92178�
!conv3d_29/StatefulPartitionedCallStatefulPartitionedCall*conv3d_28/StatefulPartitionedCall:output:0conv3d_29_92196conv3d_29_92198*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_29_layer_call_and_return_conditional_losses_92195�
add_1/PartitionedCallPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0*conv3d_24/StatefulPartitionedCall:output:0*conv3d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_92209�
!conv3d_30/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv3d_30_92222conv3d_30_92224*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_30_layer_call_and_return_conditional_losses_92221�
IdentityIdentity*conv3d_30/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   �	
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall"^conv3d_19/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall"^conv3d_20/StatefulPartitionedCall"^conv3d_21/StatefulPartitionedCall"^conv3d_22/StatefulPartitionedCall"^conv3d_23/StatefulPartitionedCall"^conv3d_24/StatefulPartitionedCall"^conv3d_25/StatefulPartitionedCall"^conv3d_26/StatefulPartitionedCall"^conv3d_27/StatefulPartitionedCall"^conv3d_28/StatefulPartitionedCall"^conv3d_29/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall"^conv3d_30/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
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
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_95244

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:���������*
ksize	
*
paddingSAME*
strides	
f
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_20_layer_call_fn_95605

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_20_layer_call_and_return_conditional_losses_91668{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_21_layer_call_and_return_conditional_losses_91836

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_95484

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
TanhTanhBiasAdd:output:0*
T0*3
_output_shapes!
:���������c
IdentityIdentityTanh:y:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_91250

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�

�
D__inference_conv3d_30_layer_call_and_return_conditional_losses_92221

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
E
)__inference_reshape_1_layer_call_fn_95561

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_91638l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_91524

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_1_layer_call_fn_95414

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_91546l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_25_layer_call_fn_95625

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_25_layer_call_and_return_conditional_losses_91651{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_27_layer_call_and_return_conditional_losses_95882

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_91286

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingSAME*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_4_layer_call_fn_95473

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_91559{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv3d_14_layer_call_fn_95513

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_91593{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv3d_29_layer_call_and_return_conditional_losses_96200

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_29_layer_call_and_return_conditional_losses_92195

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_1<
serving_default_input_1:0���������   I
	conv3d_30<
StatefulPartitionedCall:0���������   tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer-22
layer-23
layer-24
layer_with_weights-15
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-18
 layer-31
!layer_with_weights-19
!layer-32
"layer_with_weights-20
"layer-33
#layer_with_weights-21
#layer-34
$layer_with_weights-22
$layer-35
%layer_with_weights-23
%layer-36
&layer-37
'layer-38
(layer-39
)layer_with_weights-24
)layer-40
*layer_with_weights-25
*layer-41
+layer_with_weights-26
+layer-42
,layer_with_weights-27
,layer-43
-layer_with_weights-28
-layer-44
.layer_with_weights-29
.layer-45
/layer-46
0layer_with_weights-30
0layer-47
1	optimizer
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6
signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate7m�8m�=m�>m�Cm�Dm�Im�Jm�Om�Pm�Um�Vm�gm�hm�mm�nm�sm�tm�ym�zm�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�7v�8v�=v�>v�Cv�Dv�Iv�Jv�Ov�Pv�Uv�Vv�gv�hv�mv�nv�sv�tv�yv�zv�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
70
81
=2
>3
C4
D5
I6
J7
O8
P9
U10
V11
g12
h13
m14
n15
s16
t17
y18
z19
20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61"
trackable_list_wrapper
�
70
81
=2
>3
C4
D5
I6
J7
O8
P9
U10
V11
g12
h13
m14
n15
s16
t17
y18
z19
20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
+:)2conv3d/kernel
:2conv3d/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_10/kernel
:2conv3d_10/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_1/kernel
:2conv3d_1/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_6/kernel
:2conv3d_6/bias
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_11/kernel
:2conv3d_11/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_2/kernel
:2conv3d_2/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_7/kernel
:2conv3d_7/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_12/kernel
:2conv3d_12/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_3/kernel
:2conv3d_3/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_8/kernel
:2conv3d_8/bias
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_13/kernel
:2conv3d_13/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_4/kernel
:2conv3d_4/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_9/kernel
:2conv3d_9/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_14/kernel
:2conv3d_14/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_15/kernel
:2conv3d_15/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_20/kernel
:2conv3d_20/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_25/kernel
:2conv3d_25/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_16/kernel
:2conv3d_16/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_21/kernel
:2conv3d_21/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_26/kernel
:2conv3d_26/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_17/kernel
:2conv3d_17/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_22/kernel
:2conv3d_22/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_27/kernel
:2conv3d_27/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_18/kernel
:2conv3d_18/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_23/kernel
:2conv3d_23/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_28/kernel
:2conv3d_28/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_19/kernel
:2conv3d_19/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_24/kernel
:2conv3d_24/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_29/kernel
:2conv3d_29/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_30/kernel
:2conv3d_30/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
�
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
047"
trackable_list_wrapper
(
�0"
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

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
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
�2�
%__inference_model_layer_call_fn_92355
%__inference_model_layer_call_fn_93937
%__inference_model_layer_call_fn_94066
%__inference_model_layer_call_fn_93321�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_model_layer_call_and_return_conditional_losses_94585
@__inference_model_layer_call_and_return_conditional_losses_95104
@__inference_model_layer_call_and_return_conditional_losses_93496
@__inference_model_layer_call_and_return_conditional_losses_93671�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_91229input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_conv3d_layer_call_fn_95113�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_conv3d_layer_call_and_return_conditional_losses_95124�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_5_layer_call_fn_95133�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_95144�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_10_layer_call_fn_95153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_10_layer_call_and_return_conditional_losses_95164�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_1_layer_call_fn_95173�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_95184�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_6_layer_call_fn_95193�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_95204�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_11_layer_call_fn_95213�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_11_layer_call_and_return_conditional_losses_95224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_max_pooling3d_layer_call_fn_95229
-__inference_max_pooling3d_layer_call_fn_95234�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_95239
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_95244�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_max_pooling3d_2_layer_call_fn_95249
/__inference_max_pooling3d_2_layer_call_fn_95254�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_95259
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_95264�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_max_pooling3d_4_layer_call_fn_95269
/__inference_max_pooling3d_4_layer_call_fn_95274�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_95279
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_95284�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_2_layer_call_fn_95293�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_95304�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_7_layer_call_fn_95313�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_95324�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_12_layer_call_fn_95333�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_12_layer_call_and_return_conditional_losses_95344�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_3_layer_call_fn_95353�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_95364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_8_layer_call_fn_95373�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_8_layer_call_and_return_conditional_losses_95384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_13_layer_call_fn_95393�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_13_layer_call_and_return_conditional_losses_95404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_max_pooling3d_1_layer_call_fn_95409
/__inference_max_pooling3d_1_layer_call_fn_95414�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_95419
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_95424�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_max_pooling3d_3_layer_call_fn_95429
/__inference_max_pooling3d_3_layer_call_fn_95434�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_95439
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_95444�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_max_pooling3d_5_layer_call_fn_95449
/__inference_max_pooling3d_5_layer_call_fn_95454�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_95459
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_95464�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_4_layer_call_fn_95473�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_95484�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv3d_9_layer_call_fn_95493�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv3d_9_layer_call_and_return_conditional_losses_95504�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_14_layer_call_fn_95513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_14_layer_call_and_return_conditional_losses_95524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_add_layer_call_fn_95531�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_add_layer_call_and_return_conditional_losses_95539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_reshape_layer_call_fn_95544�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_reshape_layer_call_and_return_conditional_losses_95556�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_reshape_1_layer_call_fn_95561�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_reshape_1_layer_call_and_return_conditional_losses_95576�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_15_layer_call_fn_95585�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_15_layer_call_and_return_conditional_losses_95596�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_20_layer_call_fn_95605�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_20_layer_call_and_return_conditional_losses_95616�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_25_layer_call_fn_95625�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_25_layer_call_and_return_conditional_losses_95636�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_up_sampling3d_layer_call_fn_95641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_95678�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_up_sampling3d_2_layer_call_fn_95683�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_95720�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_up_sampling3d_4_layer_call_fn_95725�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_95762�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_16_layer_call_fn_95771�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_16_layer_call_and_return_conditional_losses_95782�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_21_layer_call_fn_95791�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_21_layer_call_and_return_conditional_losses_95802�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_26_layer_call_fn_95811�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_26_layer_call_and_return_conditional_losses_95822�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_17_layer_call_fn_95831�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_17_layer_call_and_return_conditional_losses_95842�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_22_layer_call_fn_95851�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_22_layer_call_and_return_conditional_losses_95862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_27_layer_call_fn_95871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_27_layer_call_and_return_conditional_losses_95882�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_up_sampling3d_1_layer_call_fn_95887�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_95948�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_up_sampling3d_3_layer_call_fn_95953�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_96014�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_up_sampling3d_5_layer_call_fn_96019�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_96080�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_18_layer_call_fn_96089�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_18_layer_call_and_return_conditional_losses_96100�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_23_layer_call_fn_96109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_23_layer_call_and_return_conditional_losses_96120�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_28_layer_call_fn_96129�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_28_layer_call_and_return_conditional_losses_96140�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_19_layer_call_fn_96149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_19_layer_call_and_return_conditional_losses_96160�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_24_layer_call_fn_96169�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_24_layer_call_and_return_conditional_losses_96180�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_29_layer_call_fn_96189�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_29_layer_call_and_return_conditional_losses_96200�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_add_1_layer_call_fn_96207�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_add_1_layer_call_and_return_conditional_losses_96215�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_30_layer_call_fn_96224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_30_layer_call_and_return_conditional_losses_96234�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_93808input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_91229�gCD=>78UVOPIJstmngh���yz��������������������������������������<�9
2�/
-�*
input_1���������   
� "A�>
<
	conv3d_30/�,
	conv3d_30���������   �
@__inference_add_1_layer_call_and_return_conditional_losses_96215����
���
���
.�+
inputs/0���������   
.�+
inputs/1���������   
.�+
inputs/2���������   
� "1�.
'�$
0���������   
� �
%__inference_add_1_layer_call_fn_96207����
���
���
.�+
inputs/0���������   
.�+
inputs/1���������   
.�+
inputs/2���������   
� "$�!���������   �
>__inference_add_layer_call_and_return_conditional_losses_95539����
���
���
.�+
inputs/0���������
.�+
inputs/1���������
.�+
inputs/2���������
� "1�.
'�$
0���������
� �
#__inference_add_layer_call_fn_95531����
���
���
.�+
inputs/0���������
.�+
inputs/1���������
.�+
inputs/2���������
� "$�!����������
D__inference_conv3d_10_layer_call_and_return_conditional_losses_95164tCD;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_10_layer_call_fn_95153gCD;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_11_layer_call_and_return_conditional_losses_95224tUV;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_11_layer_call_fn_95213gUV;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_12_layer_call_and_return_conditional_losses_95344tst;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_12_layer_call_fn_95333gst;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_13_layer_call_and_return_conditional_losses_95404v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_13_layer_call_fn_95393i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_14_layer_call_and_return_conditional_losses_95524v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_14_layer_call_fn_95513i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_15_layer_call_and_return_conditional_losses_95596v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_15_layer_call_fn_95585i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_16_layer_call_and_return_conditional_losses_95782v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_16_layer_call_fn_95771i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_17_layer_call_and_return_conditional_losses_95842v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_17_layer_call_fn_95831i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_18_layer_call_and_return_conditional_losses_96100v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_18_layer_call_fn_96089i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_19_layer_call_and_return_conditional_losses_96160v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_19_layer_call_fn_96149i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
C__inference_conv3d_1_layer_call_and_return_conditional_losses_95184tIJ;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
(__inference_conv3d_1_layer_call_fn_95173gIJ;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_20_layer_call_and_return_conditional_losses_95616v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_20_layer_call_fn_95605i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_21_layer_call_and_return_conditional_losses_95802v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_21_layer_call_fn_95791i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_22_layer_call_and_return_conditional_losses_95862v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_22_layer_call_fn_95851i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_23_layer_call_and_return_conditional_losses_96120v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_23_layer_call_fn_96109i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_24_layer_call_and_return_conditional_losses_96180v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_24_layer_call_fn_96169i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_25_layer_call_and_return_conditional_losses_95636v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_25_layer_call_fn_95625i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_26_layer_call_and_return_conditional_losses_95822v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_26_layer_call_fn_95811i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_27_layer_call_and_return_conditional_losses_95882v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
)__inference_conv3d_27_layer_call_fn_95871i��;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_28_layer_call_and_return_conditional_losses_96140v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_28_layer_call_fn_96129i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_29_layer_call_and_return_conditional_losses_96200v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_29_layer_call_fn_96189i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
C__inference_conv3d_2_layer_call_and_return_conditional_losses_95304tgh;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
(__inference_conv3d_2_layer_call_fn_95293ggh;�8
1�.
,�)
inputs���������
� "$�!����������
D__inference_conv3d_30_layer_call_and_return_conditional_losses_96234v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_30_layer_call_fn_96224i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
C__inference_conv3d_3_layer_call_and_return_conditional_losses_95364tyz;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
(__inference_conv3d_3_layer_call_fn_95353gyz;�8
1�.
,�)
inputs���������
� "$�!����������
C__inference_conv3d_4_layer_call_and_return_conditional_losses_95484v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
(__inference_conv3d_4_layer_call_fn_95473i��;�8
1�.
,�)
inputs���������
� "$�!����������
C__inference_conv3d_5_layer_call_and_return_conditional_losses_95144t=>;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
(__inference_conv3d_5_layer_call_fn_95133g=>;�8
1�.
,�)
inputs���������   
� "$�!���������   �
C__inference_conv3d_6_layer_call_and_return_conditional_losses_95204tOP;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
(__inference_conv3d_6_layer_call_fn_95193gOP;�8
1�.
,�)
inputs���������   
� "$�!���������   �
C__inference_conv3d_7_layer_call_and_return_conditional_losses_95324tmn;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
(__inference_conv3d_7_layer_call_fn_95313gmn;�8
1�.
,�)
inputs���������
� "$�!����������
C__inference_conv3d_8_layer_call_and_return_conditional_losses_95384u�;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
(__inference_conv3d_8_layer_call_fn_95373h�;�8
1�.
,�)
inputs���������
� "$�!����������
C__inference_conv3d_9_layer_call_and_return_conditional_losses_95504v��;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
(__inference_conv3d_9_layer_call_fn_95493i��;�8
1�.
,�)
inputs���������
� "$�!����������
A__inference_conv3d_layer_call_and_return_conditional_losses_95124t78;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
&__inference_conv3d_layer_call_fn_95113g78;�8
1�.
,�)
inputs���������   
� "$�!���������   �
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_95419�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_95424p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
/__inference_max_pooling3d_1_layer_call_fn_95409�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
/__inference_max_pooling3d_1_layer_call_fn_95414c;�8
1�.
,�)
inputs���������
� "$�!����������
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_95259�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_95264p;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������
� �
/__inference_max_pooling3d_2_layer_call_fn_95249�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
/__inference_max_pooling3d_2_layer_call_fn_95254c;�8
1�.
,�)
inputs���������   
� "$�!����������
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_95439�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_95444p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
/__inference_max_pooling3d_3_layer_call_fn_95429�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
/__inference_max_pooling3d_3_layer_call_fn_95434c;�8
1�.
,�)
inputs���������
� "$�!����������
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_95279�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
J__inference_max_pooling3d_4_layer_call_and_return_conditional_losses_95284p;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������
� �
/__inference_max_pooling3d_4_layer_call_fn_95269�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
/__inference_max_pooling3d_4_layer_call_fn_95274c;�8
1�.
,�)
inputs���������   
� "$�!����������
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_95459�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
J__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_95464p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
/__inference_max_pooling3d_5_layer_call_fn_95449�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
/__inference_max_pooling3d_5_layer_call_fn_95454c;�8
1�.
,�)
inputs���������
� "$�!����������
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_95239�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_95244p;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������
� �
-__inference_max_pooling3d_layer_call_fn_95229�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
-__inference_max_pooling3d_layer_call_fn_95234c;�8
1�.
,�)
inputs���������   
� "$�!����������
@__inference_model_layer_call_and_return_conditional_losses_93496�gCD=>78UVOPIJstmngh���yz��������������������������������������D�A
:�7
-�*
input_1���������   
p 

 
� "1�.
'�$
0���������   
� �
@__inference_model_layer_call_and_return_conditional_losses_93671�gCD=>78UVOPIJstmngh���yz��������������������������������������D�A
:�7
-�*
input_1���������   
p

 
� "1�.
'�$
0���������   
� �
@__inference_model_layer_call_and_return_conditional_losses_94585�gCD=>78UVOPIJstmngh���yz��������������������������������������C�@
9�6
,�)
inputs���������   
p 

 
� "1�.
'�$
0���������   
� �
@__inference_model_layer_call_and_return_conditional_losses_95104�gCD=>78UVOPIJstmngh���yz��������������������������������������C�@
9�6
,�)
inputs���������   
p

 
� "1�.
'�$
0���������   
� �
%__inference_model_layer_call_fn_92355�gCD=>78UVOPIJstmngh���yz��������������������������������������D�A
:�7
-�*
input_1���������   
p 

 
� "$�!���������   �
%__inference_model_layer_call_fn_93321�gCD=>78UVOPIJstmngh���yz��������������������������������������D�A
:�7
-�*
input_1���������   
p

 
� "$�!���������   �
%__inference_model_layer_call_fn_93937�gCD=>78UVOPIJstmngh���yz��������������������������������������C�@
9�6
,�)
inputs���������   
p 

 
� "$�!���������   �
%__inference_model_layer_call_fn_94066�gCD=>78UVOPIJstmngh���yz��������������������������������������C�@
9�6
,�)
inputs���������   
p

 
� "$�!���������   �
D__inference_reshape_1_layer_call_and_return_conditional_losses_95576e0�-
&�#
!�
inputs����������
� "1�.
'�$
0���������
� �
)__inference_reshape_1_layer_call_fn_95561X0�-
&�#
!�
inputs����������
� "$�!����������
B__inference_reshape_layer_call_and_return_conditional_losses_95556e;�8
1�.
,�)
inputs���������
� "&�#
�
0����������
� �
'__inference_reshape_layer_call_fn_95544X;�8
1�.
,�)
inputs���������
� "������������
#__inference_signature_wrapper_93808�gCD=>78UVOPIJstmngh���yz��������������������������������������G�D
� 
=�:
8
input_1-�*
input_1���������   "A�>
<
	conv3d_30/�,
	conv3d_30���������   �
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_95948p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������   
� �
/__inference_up_sampling3d_1_layer_call_fn_95887c;�8
1�.
,�)
inputs���������
� "$�!���������   �
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_95720p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
/__inference_up_sampling3d_2_layer_call_fn_95683c;�8
1�.
,�)
inputs���������
� "$�!����������
J__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_96014p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������   
� �
/__inference_up_sampling3d_3_layer_call_fn_95953c;�8
1�.
,�)
inputs���������
� "$�!���������   �
J__inference_up_sampling3d_4_layer_call_and_return_conditional_losses_95762p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
/__inference_up_sampling3d_4_layer_call_fn_95725c;�8
1�.
,�)
inputs���������
� "$�!����������
J__inference_up_sampling3d_5_layer_call_and_return_conditional_losses_96080p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������   
� �
/__inference_up_sampling3d_5_layer_call_fn_96019c;�8
1�.
,�)
inputs���������
� "$�!���������   �
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_95678p;�8
1�.
,�)
inputs���������
� "1�.
'�$
0���������
� �
-__inference_up_sampling3d_layer_call_fn_95641c;�8
1�.
,�)
inputs���������
� "$�!���������