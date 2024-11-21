#include <iostream>
#include <vector>
#include <array>
#include <cstdint>

using ui32 = uint32_t;
using ui16 = uint16_t;

const int BUCKETS = 1024;  // Размер гистограммы

// Функция для получения 10 средних битов из хеша
inline ui16 GetMiddleBits(ui32 hash) {
    return (hash >> 11) % BUCKETS; // Берем 10 битов из середины хеша
}

// Структура для хранения результатов подсчета
struct Bucket {
    ui16 count1 = 0; // Счетчик совпадений для первого массива
    ui16 count2 = 0; // Счетчик совпадений для второго массива
};

// Функция для оценки кардинальности на основе двух массивов хешей
void CalculateCardinality(const std::vector<ui32>& hashes1, const std::vector<ui32>& hashes2) {
    std::array<Bucket, BUCKETS> histogram1 = {}; // Первая гистограмма
    std::array<Bucket, BUCKETS> histogram2 = {}; // Вторая гистограмма

    // Заполняем первую гистограмму на основе первой функции хеширования
    for (ui32 hash : hashes1) {
        ui16 bucketIdx = GetMiddleBits(hash);
        histogram1[bucketIdx].count1++;
    }
    for (ui32 hash : hashes2) {
        ui16 bucketIdx = GetMiddleBits(hash);
        histogram1[bucketIdx].count2++;
    }

    // Заполняем вторую гистограмму с другой выборкой битов из хеша
    for (ui32 hash : hashes1) {
        ui16 bucketIdx = (hash >> 15) % BUCKETS; // Берем другие 10 битов
        histogram2[bucketIdx].count1++;
    }
    for (ui32 hash : hashes2) {
        ui16 bucketIdx = (hash >> 15) % BUCKETS; // Берем другие 10 битов
        histogram2[bucketIdx].count2++;
    }

    // Подсчитываем результаты
    ui64 joinCardinality = 0;

    for (const auto& bucket : histogram1) {
        joinCardinality += static_cast<ui64>(bucket.count1) * bucket.count2;
    }
    for (const auto& bucket : histogram2) {
        joinCardinality += static_cast<ui64>(bucket.count1) * bucket.count2;
    }

    std::cout << "Оценка кардинальности джоина: " << joinCardinality / 2 << std::endl;
}
