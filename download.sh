#!/bin/bash

model_name=$1

hf download $model_name
hf download Anthropic/hh-rlhf --repo-type dataset