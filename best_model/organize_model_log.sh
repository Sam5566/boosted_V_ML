#!/bin/bash
########################################
# This file remove the backspace and return characters in the log file generated by keras moduels
# so that the log file can be easy to look at, and also reduce the file size.
######################################## 

file=$1
echo "Organizing the log file: $file ..."

sed -i -r 's/.**(.{150})/\1/g' $file
sed -i -r 's/[]//g' $file

