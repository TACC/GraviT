class RandEngine {
public:
  inline float fastrand(float min, float max) {
    g_seed = 214013 * (g_seed) + 2531011;
    return min + (g_seed >> 16) * (1.0f / 65535.0f) * (max - min);
  }

  inline float fastrand(unsigned int *seedval, float min, float max) {
    *seedval = 214013 * (*seedval) + 2531011;
    return min + (*seedval >> 16) * (1.0f / 65535.0f) * (max - min);
  }

  void SetSeed(unsigned int seedval) { g_seed = seedval; }

  unsigned int *ReturnSeed() { return &g_seed; }

protected:
  unsigned int g_seed;
};
