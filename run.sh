#!/bin/bash

set -e  # Salir si hay error

BUILD_DIR="build"
MAIN_EXEC="neuralnet"

# Crear carpeta de build si no existe
if [ ! -d "$BUILD_DIR" ]; then
    echo "🔧 Creando carpeta de compilación..."
    mkdir "$BUILD_DIR"
fi

# Configurar con CMake
echo "📁 Configurando el proyecto con CMake..."
cmake -S . -B "$BUILD_DIR"

# Compilar
echo "🛠️ Compilando el proyecto..."
cmake --build "$BUILD_DIR"

# Ejecutar si hay argumentos
if [ "$#" -gt 0 ]; then
    echo "🚀 Ejecutando '$MAIN_EXEC' con argumentos: $@"
    echo "======================================="
    "$BUILD_DIR/$MAIN_EXEC" "$@"
    echo "======================================="
else
    echo "✅ Compilación finalizada."
    echo "ℹ️ Puedes ejecutar: ./run.sh <args> para correr neuralnet"
    echo "📂 Ejecutables adicionales disponibles en $BUILD_DIR:"
    ls "$BUILD_DIR"
fi
