# sources = Matrix.cpp
# executable = Matrix.so
# CXX = clang++
# CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC  `python3 -m pybind11 --includes` `python3-config --includes --ldflags`

# all: $(sources)
# 	$(CXX) $(CXXFLAGS) $(shell python3 -m pybind11 --includes) $(sources) -o $(executable)

# clear:
# 	rm -rf $(executable) __pycache__ .pytest* *.so

# test:
# 	pytest test_Matrix.py

# 編譯器設定
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -fPIC
PYBIND11_INC = $(shell python3 -m pybind11 --includes)
PYTHON_INC = $(shell python3-config --includes)
PYTHON_LDFLAGS = $(shell python3-config --ldflags)
INCLUDES = -I. $(PYBIND11_INC) $(PYTHON_INC)

# 檔案設定
TARGET = Matrix.so
SOURCES = Matrix.cpp SVD.cpp PCA.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# 預設目標
all: $(TARGET)

# 編譯 shared object
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $(OBJECTS) -o $@ $(PYTHON_LDFLAGS)

# 編譯各個 .o 檔案
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 清理
clean:
	rm -f $(OBJECTS) $(TARGET)
