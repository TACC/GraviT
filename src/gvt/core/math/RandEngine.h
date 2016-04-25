class RandEngine {
public:
#define rotl(r, n) (((r) << (n)) | ((r) >> ((8 * sizeof(r)) - (n))))

  inline float rng(uint &seed) {
    uint x, y, z;
    x = (seed >> 16) + 4125832013u;   // upper 16 bits + offset
    y = (seed & 0xffff) + 814584116u; // lower 16 bits + offset
    z = 542;
    x *= 255519323u;
    x = rotl(x, 13); // CMR, period = 4294785923 (prime)
    y *= 3166389663u;
    y = rotl(y, 17); // CMR, period = 4294315741 (prime)
    z -= rotl(z, 11);
    z = rotl(z, 27); // RSR, period = 253691 = 2^3*3^2*71*557
    seed = x ^ y ^ z;
    return ((float)(seed & 0x00FFFFFF) / (float)0x01000000);
  }

  inline float rnghost(uint &seed) {
    uint x, y, z;
    x = (seed >> 16) + 4125832013u;   // upper 16 bits + offset
    y = (seed & 0xffff) + 814584116u; // lower 16 bits + offset
    z = 542;
    x *= 255519323u;
    x = rotl(x, 13); // CMR, period = 4294785923 (prime)
    y *= 3166389663u;
    y = rotl(y, 17); // CMR, period = 4294315741 (prime)
    z -= rotl(z, 11);
    z = rotl(z, 27); // RSR, period = 253691 = 2^3*3^2*71*557
    seed = x ^ y ^ z;
    return ((float)(seed & 0x00FFFFFF) / (float)0x01000000);
  }

  inline float fastrand(float min, float max) {
    // g_seed = 214013 * (g_seed) + 2531011;
    return min + rng(g_seed) * (max - min);
  }

  inline float fastrand(unsigned int *seedval, float min, float max) {
    *seedval = 214013 * (*seedval) + 2531011;
    return min + (*seedval >> 16) * ff * (max - min);
  }

  void SetSeed(unsigned int seedval) { g_seed = seedval; }

  unsigned int *ReturnSeed() { return &g_seed; }

protected:
  unsigned int g_seed;
  const float ff = (1.0f / 65535.0f);
};
