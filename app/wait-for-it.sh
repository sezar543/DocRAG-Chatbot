#!/usr/bin/env bash
# wait-for-it.sh

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

# check if netcat is installed
if ! command -v nc &> /dev/null
then
    echo "nc (netcat) could not be found"
    exit
fi

until nc -z "$host" "$port"; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing command"
exec $cmd