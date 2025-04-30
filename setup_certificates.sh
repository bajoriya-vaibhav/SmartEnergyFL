# !/bin/bash
set -e

mkdir -p certificates
cd certificates

rm -f *.key *.crt *.pem *.srl *.csr

openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=FlowerCA"

openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=server"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 3650 -sha256

cat server.crt server.key > server.pem

for i in 1 2; do
  openssl genrsa -out client-${i}.key 2048
  openssl req -new -key client-${i}.key -out client-${i}.csr \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=client-${i}"
  openssl x509 -req -in client-${i}.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out client-${i}.crt -days 3650 -sha256
  cat client-${i}.crt client-${i}.key > client-${i}.pem
done

echo "Certificates generated successfully!"
cd ..