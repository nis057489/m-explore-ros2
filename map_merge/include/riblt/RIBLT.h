#pragma once

#include <cmath>
#include <iostream>
#include <queue>
#include <optional>
#include <functional>
#include <stdexcept>
#include "Crypto.h"
#include "ParseUtils.h"
#include <unordered_set>

namespace riblet
{
    // Add comparison operators for Hash before the CodedSymbol class
    inline bool operator==(const Hash& lhs, const Hash& rhs) {
        return std::memcmp(lhs.h, rhs.h, sizeof(lhs.h)) == 0;
    }

    inline bool operator!=(const Hash& lhs, const Hash& rhs) {
        return !(lhs == rhs);
    }

    // -----------------------
    // CodedSymbol Class
    // -----------------------
    struct CodedSymbol
    {
        Hash hash;
        int64_t count = 0;
        std::string val;

        // Add default constructor
        CodedSymbol() : hash(), count(0), val() {}

        // Existing constructor
        CodedSymbol(std::string_view valRaw, int64_t count = 1)
            : count(count) {
            val.reserve(valRaw.size() + 4);
            val += to_sv<uint32_t>(static_cast<uint32_t>(valRaw.size()));
            val += valRaw;
            hash = Hash(valRaw);
        }

        // Add copy constructor
        CodedSymbol(const CodedSymbol& other)
            : hash(other.hash)
            , count(other.count)
            , val(other.val) {}

        // Keep existing move constructor
        CodedSymbol(CodedSymbol&& other) noexcept
            : hash(other.hash)
            , count(other.count)
            , val(std::move(other.val)) {}

        // Add copy assignment operator
        CodedSymbol& operator=(const CodedSymbol& other) {
            if (this != &other) {
                hash = other.hash;
                count = other.count;
                val = other.val;
            }
            return *this;
        }

        // Add move assignment operator
        CodedSymbol& operator=(CodedSymbol&& other) noexcept {
            if (this != &other) {
                hash = other.hash;
                count = other.count;
                val = std::move(other.val);
            }
            return *this;
        }

        // Existing constructor
        CodedSymbol(std::string_view val, Hash hash, int64_t count)
            : hash(hash), count(count), val(val) {}

        void add(const CodedSymbol &other)
        {
            update(other, other.count);
        }

        void sub(const CodedSymbol &other)
        {
            update(other, -other.count);
        }

        void negate()
        {
            count *= -1;
        }

        bool isPure() const
        {
            if (count == 1 || count == -1)
            {
                auto v = decodeVal();
                if (!v)
                    return false;
                return Hash(*v) == hash;
            }
            return false;
        }

        bool isZero() const
        {
            static const Hash emptyHash;
            return count == 0 && hash == emptyHash;
        }

        std::string getVal() const
        {
            auto v = decodeVal();
            if (!v)
                throw std::runtime_error("val not decodeable (not pure?)");
            return std::string(*v);
        }

        std::string_view getHashSV() const
        {
            return std::string_view(reinterpret_cast<const char *>(hash.h), sizeof(hash.h));
        }

    private:
        // Optimize XOR operations by using SIMD where possible
        void update(const CodedSymbol& other, int64_t inc) {
            if (other.val.size() > val.size())
                val.resize(other.val.size(), '\0');

            // Use vectorized XOR operations for performance
            const size_t val_size = other.val.size();
            const char* src = other.val.data();
            char* dst = val.data();
            
            #if defined(__AVX2__)
                // Use AVX2 instructions for faster XOR
                for (size_t i = 0; i + 32 <= val_size; i += 32) {
                    __m256i a = _mm256_loadu_si256((__m256i*)(dst + i));
                    __m256i b = _mm256_loadu_si256((__m256i*)(src + i));
                    _mm256_storeu_si256((__m256i*)(dst + i), _mm256_xor_si256(a, b));
                }
            #endif

            // Handle remaining bytes
            for (size_t i = (val_size / 32) * 32; i < val_size; i++) {
                dst[i] ^= src[i];
            }

            // Optimize hash XOR
            for (size_t i = 0; i < sizeof(hash.h); i++) {
                hash.h[i] ^= other.hash.h[i];
            }

            count += inc;
        }

        std::optional<std::string_view> decodeVal() const
        {
            auto v = std::string_view(val);

            if (v.size() < 4)
                throw std::runtime_error("val unexpectedly small");
            auto size = from_sv<uint32_t>(v.substr(0, 4));
            v = v.substr(4);
            if (size > v.size())
                return {};

            return v.substr(0, size);
        }
    };

    // -----------------------
    // IndexGenerator Class
    // -----------------------
    struct IndexGenerator
    {
        PRNG prng;
        uint64_t curr = 0;

        IndexGenerator(const CodedSymbol &sym) : prng(sym.hash.getPRNG()) {}

        void jump()
        {
            uint64_t rand = prng.rand64();
            // Magic formula from the RIBLT paper:
            curr += uint64_t(ceil((double(curr) + 1.5) *
                                  ((uint64_t(1) << 32) / std::sqrt(double(rand) + 1) - 1)));
        }
    };

    // -----------------------
    // SymbolQueue Class
    // -----------------------
    struct SymbolQueue
    {
        struct QueueEntry
        {
            uint64_t codedStreamIndex;
            uint64_t symbolIndex;

            QueueEntry(uint64_t streamIndex, uint64_t symIndex)
                : codedStreamIndex(streamIndex), symbolIndex(symIndex) {}

            QueueEntry() = default;

            friend bool operator>(QueueEntry const &a, QueueEntry const &b)
            {
                return a.codedStreamIndex > b.codedStreamIndex;
            }
        };

        struct SymbolEntry
        {
            IndexGenerator gen;
            CodedSymbol sym;

            SymbolEntry(const IndexGenerator &generator, CodedSymbol symbol)
                : gen(generator), sym(std::move(symbol)) {}

            SymbolEntry() = default;
        };

        std::vector<SymbolEntry> symbolVec;
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> codedQueue;

        void enqueue(CodedSymbol &&sym, const IndexGenerator &gen)
        {
            codedQueue.emplace(gen.curr, symbolVec.size());
            symbolVec.emplace_back(gen, std::move(sym));
        }

        uint64_t nextCodedIndex() const
        {
            if (symbolVec.empty())
                return std::numeric_limits<uint64_t>::max();
            return codedQueue.top().codedStreamIndex;
        }

        void bubbleUp()
        {
            if (symbolVec.empty())
                throw std::runtime_error("can't bubbleUp empty queue");
            auto e = codedQueue.top();
            codedQueue.pop();
            e.codedStreamIndex = symbolVec[e.symbolIndex].gen.curr;
            codedQueue.push(e);
        }

        void clear()
        {
            symbolVec.clear();
            symbolVec.shrink_to_fit();
            decltype(codedQueue) emptyQueue;
            codedQueue.swap(emptyQueue);
        }
    };

    // -----------------------
    // RIBLT Class
    // -----------------------
    struct RIBLT
    {
        // Pre-allocate vectors to avoid resizing
        std::vector<CodedSymbol> codedSymbols;
        SymbolQueue queue;
        bool doneExpanding = false;
        size_t nextPeel = 0;

        void expand(size_t n)
        {
            if (doneExpanding)
                throw std::runtime_error("can't expand fixed size vector");
            
            // Reserve space to avoid reallocation
            codedSymbols.reserve(codedSymbols.size() + n);
            size_t oldSize = codedSymbols.size();
            codedSymbols.resize(oldSize + n);

            // Batch process queued symbols
            while (queue.nextCodedIndex() < codedSymbols.size())
            {
                auto &top = queue.symbolVec[queue.codedQueue.top().symbolIndex];
                diffuseSymbol(top.sym, top.gen);
                queue.bubbleUp();
            }
        }

        void push(const CodedSymbol &sym)
        {
            if (codedSymbols.empty()) {
                expand(1);
            }
            // Distribute the symbol across all positions
            IndexGenerator gen(sym);
            diffuseSymbol(sym, gen);
        }

        void setDoneExpanding()
        {
            doneExpanding = true;
            queue.clear();
        }

        void add(CodedSymbol &&sym, const std::function<void(size_t)> &onSym = [](size_t) {})
        {
            if (codedSymbols.empty()) {
                expand(1);
            }
            IndexGenerator gen(sym);
            diffuseSymbol(sym, gen, onSym);
            if (!doneExpanding) {
                queue.enqueue(std::move(sym), gen);
            }
        }

        void peel(const std::function<void(const CodedSymbol &sym)> &onSym)
        {
            std::vector<size_t> decodeable;
            std::unordered_set<size_t> processed;

            // First pass: find initial pure symbols
            for (size_t i = 0; i < codedSymbols.size(); i++) {
                auto& symbol = codedSymbols[i];
                if (symbol.isPure() && !processed.count(i)) {
                    decodeable.push_back(i);
                }
            }

            // Iteratively decode and remove symbols
            while (!decodeable.empty()) {
                size_t index = decodeable.back();
                decodeable.pop_back();

                if (processed.count(index))
                    continue;

                auto& symbol = codedSymbols[index];
                if (!symbol.isPure() || symbol.isZero())
                    continue;

                // Process this pure symbol
                onSym(symbol);
                processed.insert(index);

                // Use this symbol to decode others
                IndexGenerator gen(symbol);
                while (gen.curr < codedSymbols.size()) {
                    size_t curr = gen.curr;
                    if (curr != index && !processed.count(curr)) {
                        // Remove this symbol's contribution from other locations
                        codedSymbols[curr].sub(symbol);
                        
                        // If we created a new pure symbol, add it to decode list
                        if (codedSymbols[curr].isPure()) {
                            decodeable.push_back(curr);
                        }
                    }
                    gen.jump();
                }

                // Zero out the used symbol
                codedSymbols[index] = CodedSymbol();
            }
        }

        bool isBalanced()
        {
            if (codedSymbols.empty()) return true;
            // A balanced RIBLT should have all symbols properly distributed
            for (const auto& symbol : codedSymbols) {
                if (!symbol.isZero()) return false;
            }
            return true;
        }

    private:
        void diffuseSymbol(const CodedSymbol &sym, IndexGenerator &gen, const std::function<void(size_t)> &cb = [](size_t) {})
        {
            size_t initialSize = codedSymbols.size();
            while (gen.curr < initialSize)
            {
                codedSymbols[gen.curr].add(sym);
                cb(gen.curr);
                gen.jump();
            }
        }
    };

} // namespace riblet

// -----------------------
// Type Alias
// -----------------------
using RIBLT = riblet::RIBLT;
