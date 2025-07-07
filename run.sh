#!/bin/bash

set -e  # Salir si hay error
export OMP_NUM_THREADS=12   # 👈 Fuerza OpenMP a usar 12 hilos

BUILD_DIR="build"
MAIN_EXEC="neuralnet"
TEST_EXEC_PREFIX="test_"  # Prefijo para ejecutables de prueba

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [opción] [argumentos...]"
    echo ""
    echo "Opciones:"
    echo "  build       Recompila el proyecto completo (limpia y construye)"
    echo "  main        Compila solo el ejecutable principal ($MAIN_EXEC)"
    echo "  test <ejecutable> [args...]  Ejecuta un programa de prueba (ej: test_cnn)"
    echo "  <args...>   Ejecuta $MAIN_EXEC con los argumentos dados"
    echo ""
    echo "Ejemplos:"
    echo "  $0 build           # Recompila todo"
    echo "  $0 main            # Compila solo $MAIN_EXEC"
    echo "  $0 test test_cnn   # Ejecuta el test 'test_cnn'"
    echo "  $0 arg1 arg2       # Ejecuta $MAIN_EXEC con argumentos"
}

# Crear carpeta de build si no existe
ensure_build_dir() {
    if [ ! -d "$BUILD_DIR" ]; then
        echo "🔧 Creando carpeta de compilación..."
        mkdir -p "$BUILD_DIR"
    fi
}

# Configurar con CMake
configure_project() {
    echo "📁 Configurando el proyecto con CMake..."
    cmake -S . -B "$BUILD_DIR" -DCMAKE_CXX_FLAGS="-fopenmp -O3"
}

# Compilar el proyecto completo
build_all() {
    ensure_build_dir
    configure_project
    echo "🛠️ Compilando el proyecto completo..."
    cmake --build "$BUILD_DIR"
}

# Compilar solo el ejecutable principal
# Compilar solo el ejecutable principal y ejecutarlo
build_main() {
    ensure_build_dir
    configure_project
    echo "🛠️ Compilando solo $MAIN_EXEC..."
    cmake --build "$BUILD_DIR" --target "$MAIN_EXEC"
    
    # Ejecutar automáticamente después de compilar
    if [ -f "$BUILD_DIR/$MAIN_EXEC" ]; then
        echo "🚀 Ejecutando $MAIN_EXEC..."
        echo "======================================="
        "$BUILD_DIR/$MAIN_EXEC"
        echo "======================================="
    else
        echo "❌ Error: No se encontró el ejecutable $BUILD_DIR/$MAIN_EXEC"
        exit 1
    fi
}

# Ejecutar el programa principal con argumentos
run_main() {
    echo "🚀 Ejecutando '$MAIN_EXEC' con argumentos: $@"
    echo "======================================="
    "$BUILD_DIR/$MAIN_EXEC" "$@"
    echo "======================================="
}

# Ejecutar un programa de prueba
run_test() {
    local test_exec="$1"
    shift  # Remover el primer argumento (el nombre del test)
    
    if [ -z "$test_exec" ]; then
        echo "❌ Error: Debes especificar un ejecutable de prueba."
        show_help
        exit 1
    fi

    local test_path="$BUILD_DIR/$test_exec"
    if [ ! -f "$test_path" ]; then
        echo "❌ Error: El ejecutable '$test_exec' no existe en $BUILD_DIR."
        echo "📂 Ejecutables disponibles:"
        ls "$BUILD_DIR" | grep "^${TEST_EXEC_PREFIX}"
        exit 1
    fi

    echo "🧪 Ejecutando prueba '$test_exec' con argumentos: $@"
    echo "======================================="
    "$test_path" "$@"
    echo "======================================="
}

# --- Lógica principal del script ---

case "$1" in
    "build")
        # Limpia y recompila todo
        if [ -d "$BUILD_DIR" ]; then
            echo "🧹 Limpiando compilación previa..."
            rm -rf "$BUILD_DIR"
        fi
        build_all
        ;;
    "main")
        # Compila solo el ejecutable principal
        build_main
        ;;
    "test")
        # Ejecuta un test con argumentos
        shift  # Remover "test" de los argumentos
        run_test "$@"
        ;;
    "-h"|"--help")
        show_help
        ;;
    *)
        if [ "$#" -gt 0 ]; then
            # Si hay argumentos pero no son opciones conocidas, ejecuta el programa principal
            run_main "$@"
        else
            # Sin argumentos: muestra estado y ayuda
            echo "✅ Compilación finalizada."
            echo ""
            echo "ℹ️  Ejecutables disponibles en $BUILD_DIR:"
            ls "$BUILD_DIR"
            echo ""
            show_help
        fi
        ;;
esac