# 編譯器設定
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -fPIC -mavx -mfma
PYBIND11_INC = $(shell python3 -m pybind11 --includes)
PYTHON_INC = $(shell python3-config --includes)
PYTHON_LDFLAGS = $(shell python3-config --ldflags)
INCLUDES = -I. $(PYBIND11_INC) $(PYTHON_INC)

# 檔案設定
SRCDIR = src
INCDIR = include
OBJDIR = obj
TARGET = python/Matrix.so
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

# 預設目標
all: $(TARGET)

# 編譯 shared object
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $(OBJECTS) -o $@ $(PYTHON_LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/%.hpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
clean:
	rm -rf $(OBJDIR) $(TARGET)
