#ifndef PARSE_COMMAND_LINE_H
#define PARSE_COMMAND_LINE_H

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

struct ParseCommandLine {

  std::map<std::string, unsigned long> _option;
  std::map<std::string, std::string> _desc;
  std::map<std::string, std::string> _value;
  std::map<std::string, std::set<std::string> > _conflicts;
  std::map<std::string, std::set<std::string> > _requires;

  // int argc;
  // char *argv[];

  enum OPTION_TYPE { NONE = 0x0, INT, LONG, FLOAT, DOUBLE, STRING, PATH, HELP };
  const static unsigned long REQUIRED = 0x80;
  std::string appname;

  ParseCommandLine(std::string appname) : appname(appname) {
    _option["h"] = (HELP << 4) | 0;
    _desc["h"] = "Help";
  }

  void addoption(std::string name, unsigned long type, std::string desc = "", unsigned long size = 1) {
    if (_option.find(name) == _option.end()) {
      _option[name] = (type << 4) | size;
      _desc[name] = desc + ((type & REQUIRED) ? " (REQUIRED)" : "");
    }
  }

  void addconflict(std::string op1, std::string op2) {
    _conflicts[op2].insert(op1);
    _conflicts[op1].insert(op2);
  }

  bool conflict(std::string op, std::string &conf) {
    if (_conflicts.find(op) != _conflicts.end()) {
      for (auto &c : _conflicts[op]) {
        if (_value.find(c) != _value.end()) {
          conf = c;
          return true;
        }
      }
    }
    return false;
  }

  void addrequire(std::string op1, std::string op2) { _requires[op1].insert(op2); }

  bool requires(std::string op, std::string &conf) {
    if (_requires.find(op) != _requires.end()) {
      for (auto &c : _requires[op]) {
        if (_value.find(c) == _value.end()) {
          conf = c;
          return true;
        }
      }
    }
    return false;
  }

  std::string fromType(unsigned long type) {
    switch ((type >> 4) & 0x7F) {
    case 0:
      return "";
    case 1:
      return "int";
    case 2:
      return "long";
    case 3:
      return "float";
    case 4:
      return "double";
    case 5:
      return "string";
    case 6:
      return "path";
    default:
      return "unknown";
    }
  }

  void usage() {
    std::cout << appname << " usage : " << std::endl;
    for (auto &op : _option) {
      std::cout << "\t";
      std::cout << "-" << op.first << " ";
      unsigned size = op.second & 0xF;
      if (size > 0) {
        std::string tt = fromType(op.second);
        std::cout << "<";
        for (int k = 0; k < size - 1; k++) std::cout << tt << ",";
        std::cout << tt << ">";
      }

      std::cout << "\t: " << _desc[op.first] << std::endl;
      if (!_requires[op.first].empty()) {
        std::cout << "\t\t\t requires: ";
        for (auto &r : _requires[op.first]) std::cout << r << " ";
        std::cout << std::endl;
      }
      if (!_conflicts[op.first].empty()) {
        std::cout << "\t\t\t conflict: ";
        for (auto &c : _conflicts[op.first]) std::cout << c << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  int find(const int &argc, char *argv[], std::string name) {
    for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == ("-" + name)) return i;
    }
    return -1;
  }

  bool isOption(std::string name) { return (_option.find(name.substr(1, name.length())) != _option.end()); }
  bool isSet(std::string name) { return (_value.find(name) != _value.end()); }

  void parse(const int &argc, char *argv[]) {
    for (auto &op : _option) {
      int loc = find(argc, argv, op.first);

      unsigned required = ((op.second >> 4) & REQUIRED);
      if (op.first == "h" && loc != -1) {
        usage();
        exit(0);
      }

      if (required && loc == -1) {
        std::cout << "Error: option " << op.first << " is required" << std::endl << std::endl;
        usage();
        exit(0);
      }

      if (loc == -1) continue;
      std::string cwith;

      if (conflict(op.first, cwith)) {
        std::cout << "Error: option " << op.first << " conflicts with " << cwith << std::endl << std::endl;
        usage();
        exit(0);
      }

      std::string value;

      unsigned size = op.second & 0xF;

      if (size > 0) {
        if ((loc + 1) < argc && !isOption(std::string(argv[loc + 1])) && count(std::string(argv[loc + 1])) == size) {
          _value[op.first] = std::string(argv[loc + 1]);
        } else {
          std::cout << "Error: option " << op.first << " requires " << size << " values" << std::endl << std::endl;
          usage();
          exit(0);
        }
      } else {
        _value[op.first] = "";
      }
    }

    for (auto &op : _value) {
      std::string cwith;
      if (requires(op.first, cwith)) {
        std::cout << "Error: option " << op.first << " requires " << cwith << std::endl << std::endl;
        usage();
        exit(0);
      }
    }
  }

  unsigned long optionType(std::string name) {
    if (_option.find(name) != _option.end()) {
      return _option[name] >> 4;
    }
    return 0;
  }

  unsigned long optionSize(std::string name) {
    if (_option.find(name) != _option.end()) {
      return _option[name] & 0xF;
    }
    return 0;
  }
  template <typename T> std::vector<T> getValue(const std::string name) { return std::vector<T>(); };

  template <typename T> T get(const std::string name) { return; };

  int count(std::string value) {
    std::istringstream f(value);
    std::string s;
    unsigned count = 0;
    while (getline(f, s, ',')) {
      count++;
    }
    return count;
  };
};

template <> std::string ParseCommandLine::get<std::string>(const std::string name) { return _value[name]; };
template <> int ParseCommandLine::get<int>(const std::string name) { return std::stoi(_value[name]); };
template <> long ParseCommandLine::get<long>(const std::string name) { return std::stol(_value[name]); };
template <> float ParseCommandLine::get<float>(const std::string name) { return std::stof(_value[name]); };
template <> double ParseCommandLine::get<double>(const std::string name) { return std::stod(_value[name]); };

template <> std::vector<std::string> ParseCommandLine::getValue<std::string>(std::string name) {
  std::vector<std::string> value;
  if (_value.find(name) == _value.end()) return (value);
  unsigned size = _option[name] & 0xF;
  std::istringstream f(_value[name]);
  std::string s;
  unsigned count = 0;
  while (getline(f, s, ',')) {
    count++;
    value.push_back(s);
  }
  return value;
};

template <> std::vector<int> ParseCommandLine::getValue<int>(std::string name) {
  std::vector<int> value;
  if (_value.find(name) == _value.end()) return (value);
  unsigned size = _option[name] & 0xF;
  std::istringstream f(_value[name]);
  std::string s;
  unsigned count = 0;
  while (getline(f, s, ',')) {
    count++;
    value.push_back(std::stoi(s));
  }
  return (value);
};

template <> std::vector<long> ParseCommandLine::getValue<long>(std::string name) {
  std::vector<long> value;
  if (_value.find(name) == _value.end()) return (value);
  unsigned size = _option[name] & 0xF;
  std::istringstream f(_value[name]);
  std::string s;
  unsigned count = 0;
  while (getline(f, s, ',')) {
    count++;
    value.push_back(std::stol(s));
  }
  return (value);
};

template <> std::vector<float> ParseCommandLine::getValue<float>(std::string name) {
  std::vector<float> value;
  if (_value.find(name) == _value.end()) return (value);
  unsigned size = _option[name] & 0xF;
  std::istringstream f(_value[name]);
  std::string s;
  unsigned count = 0;
  while (getline(f, s, ',')) {
    count++;
    value.push_back(std::stof(s));
  }
  return (value);
};

template <> std::vector<double> ParseCommandLine::getValue<double>(std::string name) {
  std::vector<double> value;
  if (_value.find(name) == _value.end()) return (value);
  unsigned size = _option[name] & 0xF;
  std::istringstream f(_value[name]);
  std::string s;
  unsigned count = 0;
  while (getline(f, s, ',')) {
    count++;
    value.push_back(std::stod(s));
  }
  return (value);
};

#endif /*PARSE_COMMAND_LINE_H*/
