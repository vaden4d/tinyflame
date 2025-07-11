#!/bin/bash

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 --type | --style | --test"
    exit 1
fi

case $1 in
    --style)
        flake8 . --ignore E501
        ;;
    --type)
        mypy .
        ;;
    --test)
        pytest
        ;;
    *)
        echo "Unknown parameter passed: $1"
        echo "Usage: $0 --type | --style | --test"
        exit 1
        ;;
esac