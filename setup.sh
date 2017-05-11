#! /bin/bash

# Set up the tables
psql -U $USER -f model/create.sql
