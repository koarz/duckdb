//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/data_chunk.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/winapi.hpp"

namespace duckdb {
class Allocator;
class ClientContext;
class ExecutionContext;
class VectorCache;
class Serializer;
class Deserializer;

//!  A Data Chunk represents a set of vectors.
/*!
    The data chunk class is the intermediate representation used by the
   execution engine of DuckDB. It effectively represents a subset of a relation.
   It holds a set of vectors that all have the same length.

    DataChunk is initialized using the DataChunk::Initialize function by
   providing it with a vector of TypeIds for the Vector members. By default,
   this function will also allocate a chunk of memory in the DataChunk for the
   vectors and all the vectors will be referencing vectors to the data owned by
   the chunk. The reason for this behavior is that the underlying vectors can
   become referencing vectors to other chunks as well (i.e. in the case an
   operator does not alter the data, such as a Filter operator which only adds a
   selection vector).

    In addition to holding the data of the vectors, the DataChunk also owns the
   selection vector that underlying vectors can point to.
*/
//! DataChunk（数据块）表示一组向量。
/*!
    DataChunk 类是 DuckDB 执行引擎中使用的中间表示。
    从语义上看，它等价于一个关系（表）的一个子集。

    它内部持有一组向量，这些向量的长度始终保持一致。

    DataChunk 通过调用 DataChunk::Initialize 函数进行初始化，
    初始化时需要提供一个 TypeId 向量，用于指定各个 Vector 成员的数据类型。
    默认情况下，该函数还会为 DataChunk 中的向量分配一块内存，
    并且所有向量都会作为引用向量（referencing vectors），
    指向由该 DataChunk 自身所拥有的数据。

    之所以采用这种设计，是因为底层向量在后续执行过程中
    可能会变成指向其他 DataChunk 的引用向量
    （例如，当算子并不修改数据时，
    比如 Filter 算子仅仅增加了一个选择向量）。

    除了保存向量数据本身之外，
    DataChunk 还拥有（own）选择向量（selection vector），
    底层向量可以指向该选择向量以实现行级过滤。
*/
class DataChunk {
public:
	//! Creates an empty DataChunk
	//! 创建一个空的 DataChunk
	DUCKDB_API DataChunk();
	DUCKDB_API ~DataChunk();

	//! The vectors owned by the DataChunk.
	//! DataChunk 拥有的向量。
	vector<Vector> data;

public:
	inline idx_t size() const { // NOLINT
		return count;
	}
	inline idx_t ColumnCount() const {
		return data.size();
	}
	inline void SetCardinality(idx_t count_p) {
		D_ASSERT(count_p <= capacity);
		this->count = count_p;
	}
	inline void SetCardinality(const DataChunk &other) {
		SetCardinality(other.size());
	}
	inline idx_t GetCapacity() const {
		return capacity;
	}
	inline void SetCapacity(idx_t capacity_p) {
		this->capacity = capacity_p;
	}
	inline void SetCapacity(const DataChunk &other) {
		SetCapacity(other.capacity);
	}

	DUCKDB_API Value GetValue(idx_t col_idx, idx_t index) const;
	DUCKDB_API void SetValue(idx_t col_idx, idx_t index, const Value &val);

	idx_t GetAllocationSize() const;

	//! Returns true if all vectors in the DataChunk are constant
	//! 如果 DataChunk 中的所有向量都是常量，则返回 true
	DUCKDB_API bool AllConstant() const;

	//! Set the DataChunk to reference another data chunk
	//! 将此 DataChunk 设置为引用另一个 DataChunk
	DUCKDB_API void Reference(DataChunk &chunk);
	//! Set the DataChunk to own the data of data chunk, destroying the other chunk in the process
	//! 将此 DataChunk 设置为拥有另一个 DataChunk 的数据，并在此过程中销毁另一个 DataChunk
	DUCKDB_API void Move(DataChunk &chunk);

	//! Initializes a DataChunk with the given types and without any vector data allocation.
	//! 使用给定的类型初始化 DataChunk，但不分配任何向量数据。
	DUCKDB_API void InitializeEmpty(const vector<LogicalType> &types);

	//! Initializes a DataChunk with the given types. Then, if the corresponding boolean in the initialize-vector is
	//! true, it initializes the vector for that data type.
	//! 使用给定的类型初始化 DataChunk。如果初始化向量中对应的布尔值为 true，
	//! 则为该数据类型初始化向量。
	DUCKDB_API void Initialize(ClientContext &context, const vector<LogicalType> &types,
	                           idx_t capacity = STANDARD_VECTOR_SIZE);
	DUCKDB_API void Initialize(Allocator &allocator, const vector<LogicalType> &types,
	                           idx_t capacity = STANDARD_VECTOR_SIZE);
	DUCKDB_API void Initialize(ClientContext &context, const vector<LogicalType> &types, const vector<bool> &initialize,
	                           idx_t capacity = STANDARD_VECTOR_SIZE);
	DUCKDB_API void Initialize(Allocator &allocator, const vector<LogicalType> &types, const vector<bool> &initialize,
	                           idx_t capacity = STANDARD_VECTOR_SIZE);

	//! Append the other DataChunk to this one. The column count and types of
	//! the two DataChunks have to match exactly. Throws an exception if there
	//! is not enough space in the chunk and resize is not allowed.
	//! 将另一个 DataChunk 追加到当前 DataChunk。两个 DataChunk 的列数和类型必须完全匹配。
	//! 如果块中没有足够的空间且不允许调整大小，则抛出异常。
	DUCKDB_API void Append(const DataChunk &other, bool resize = false, SelectionVector *sel = nullptr,
	                       idx_t count = 0);

	//! Destroy all data and columns owned by this DataChunk
	//! 销毁此 DataChunk 拥有的所有数据和列
	DUCKDB_API void Destroy();

	//! Copies the data from this chunk to another chunk.
	//! 将此块的数据复制到另一个块。
	DUCKDB_API void Copy(DataChunk &other, idx_t offset = 0) const;
	DUCKDB_API void Copy(DataChunk &other, const SelectionVector &sel, const idx_t source_count,
	                     const idx_t offset = 0) const;

	//! Splits the DataChunk in two
	//! 将 DataChunk 一分为二
	DUCKDB_API void Split(DataChunk &other, idx_t split_idx);

	//! Fuses a DataChunk onto the right of this one, and destroys the other. Inverse of Split.
	//! 将一个 DataChunk 融合到此 DataChunk 的右侧，并销毁另一个。与 Split 相反。
	DUCKDB_API void Fuse(DataChunk &other);

	//! Makes this DataChunk reference the specified columns in the other DataChunk
	//! 使此 DataChunk 引用另一个 DataChunk 中的指定列
	DUCKDB_API void ReferenceColumns(DataChunk &other, const vector<column_t> &column_ids);

	//! Turn all the vectors from the chunk into flat vectors
	//! 将块中的所有向量转换为平坦向量
	DUCKDB_API void Flatten();

	// FIXME: this is DUCKDB_API, might need conversion back to regular unique ptr?
	// 待修复：这是 DUCKDB_API，可能需要转换回常规的 unique ptr？
	DUCKDB_API unsafe_unique_array<UnifiedVectorFormat> ToUnifiedFormat();

	DUCKDB_API void Slice(const SelectionVector &sel_vector, idx_t count);

	//! Slice all Vectors from other.data[i] to data[i + 'col_offset']
	//! Turning all Vectors into Dictionary Vectors, using 'sel'
	//! 将所有向量从 other.data[i] 切片到 data[i + 'col_offset']
	//! 使用 'sel' 将所有向量转换为字典向量
	DUCKDB_API void Slice(const DataChunk &other, const SelectionVector &sel, idx_t count, idx_t col_offset = 0);

	//! Slice a DataChunk from "offset" to "offset + count"
	//! 将 DataChunk 从 "offset" 切片到 "offset + count"
	DUCKDB_API void Slice(idx_t offset, idx_t count);

	//! Resets the DataChunk to its state right after the DataChunk::Initialize
	//! function was called. This sets the count to 0, the capacity to initial_capacity and resets each member
	//! Vector to point back to the data owned by this DataChunk.
	//! 将 DataChunk 重置为调用 DataChunk::Initialize 函数后的状态。
	//! 这会将 count 设置为 0，将 capacity 设置为 initial_capacity，
	//! 并将每个成员向量重置为指向此 DataChunk 拥有的数据。
	DUCKDB_API void Reset();

	DUCKDB_API void Serialize(Serializer &serializer, bool compressed_serialization = true) const;
	DUCKDB_API void Deserialize(Deserializer &source);

	//! Hashes the DataChunk to the target vector
	//! 将 DataChunk 哈希到目标向量
	DUCKDB_API void Hash(Vector &result);
	//! Hashes specific vectors of the DataChunk to the target vector
	//! 将 DataChunk 的特定向量哈希到目标向量
	DUCKDB_API void Hash(vector<idx_t> &column_ids, Vector &result);

	//! Returns a list of types of the vectors of this data chunk
	//! 返回此 DataChunk 的向量的类型列表
	DUCKDB_API vector<LogicalType> GetTypes() const;

	//! Converts this DataChunk to a printable string representation
	//! 将此 DataChunk 转换为可打印的字符串表示
	DUCKDB_API string ToString() const;
	DUCKDB_API void Print() const;

	DataChunk(const DataChunk &) = delete;

	//! Verify that the DataChunk is in a consistent, not corrupt state. DEBUG
	//! FUNCTION ONLY!
	//! 验证 DataChunk 处于一致且未损坏的状态。
	//! 仅限调试函数！
	DUCKDB_API void Verify();

private:
	//! The amount of tuples stored in the data chunk
	//! 数据块中存储的元组数量
	idx_t count;
	//! The amount of tuples that can be stored in the data chunk
	//! 数据块中可以存储的元组数量
	idx_t capacity;
	//! The initial capacity of this chunk set during ::Initialize, used when resetting
	//! 在 ::Initialize 期间设置的此块的初始容量，用于重置时使用
	idx_t initial_capacity;
	//! Vector caches, used to store data when ::Initialize is called
	//! 向量缓存，用于在调用 ::Initialize 时存储数据
	vector<VectorCache> vector_caches;
};
} // namespace duckdb
