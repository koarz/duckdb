#include "duckdb/common/types/vector_cache.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

// VectorCacheBuffer: 向量缓存缓冲区类
// 继承自 VectorBuffer，用于缓存和重用 Vector 对象的内存和元数据
// 这样可以避免频繁的内存分配和释放，提高性能
class VectorCacheBuffer : public VectorBuffer {
public:
	// 构造函数: 根据逻辑类型创建缓存缓冲区
	// @param allocator: 内存分配器
	// @param type_p: 向量的逻辑类型
	// @param capacity_p: 向量容量，默认为标准向量大小(通常是 2048)
	explicit VectorCacheBuffer(Allocator &allocator, const LogicalType &type_p, idx_t capacity_p = STANDARD_VECTOR_SIZE)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), type(type_p), capacity(capacity_p) {
		// 获取物理存储类型（底层实际使用的类型）
		auto internal_type = type.InternalType();
		// 根据不同的物理类型，初始化不同的存储结构
		switch (internal_type) {
		case PhysicalType::LIST: {
			// LIST 类型: 动态长度的列表
			// 为列表的偏移量数组分配内存
			// 偏移量数组用于记录每个列表元素在子数据中的起始位置
			owned_data = allocator.Allocate(capacity * GetTypeIdSize(internal_type));
			// 获取列表的子元素类型
			auto &child_type = ListType::GetChildType(type);
			// 为子元素创建缓存（子元素也可能是复杂类型）
			child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type, capacity));
			// 创建子向量用于存储列表的实际元素
			auto child_vector = make_uniq<Vector>(child_type, false, false);
			// 创建 VectorListBuffer 辅助结构来管理列表数据
			auxiliary = make_shared_ptr<VectorListBuffer>(std::move(child_vector));
			break;
		}
		case PhysicalType::ARRAY: {
			// ARRAY 类型: 固定长度的数组
			// 获取数组的子元素类型
			auto &child_type = ArrayType::GetChildType(type);
			// 获取数组的固定大小
			auto array_size = ArrayType::GetSize(type);
			// 为子元素创建缓存，总容量 = 数组大小 × 向量容量
			// 例如：ARRAY[INT, 3] 类型，如果 capacity 是 2048，则需要 3*2048 个 INT 的空间
			child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type, array_size * capacity));
			// 创建子向量，大小为 array_size * capacity
			auto child_vector = make_uniq<Vector>(child_type, true, false, array_size * capacity);
			// 创建 VectorArrayBuffer 辅助结构来管理数组数据
			auxiliary = make_shared_ptr<VectorArrayBuffer>(std::move(child_vector), array_size, capacity);
			break;
		}
		case PhysicalType::STRUCT: {
			// STRUCT 类型: 结构体，包含多个命名字段
			// 获取结构体的所有子字段类型
			auto &child_types = StructType::GetChildTypes(type);
			// 为每个子字段创建缓存
			for (auto &child_type : child_types) {
				// child_type.second 是字段的类型，child_type.first 是字段名
				child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type.second, capacity));
			}
			// 创建 VectorStructBuffer 辅助结构来管理结构体数据
			auto struct_buffer = make_shared_ptr<VectorStructBuffer>(type);
			auxiliary = std::move(struct_buffer);
			break;
		}
		default:
			// 其他基础类型（INT, DOUBLE, VARCHAR 等）
			// 直接分配对应大小的内存
			owned_data = allocator.Allocate(capacity * GetTypeIdSize(internal_type));
			break;
		}
	}

	// ResetFromCache: 从缓存中重置 Vector 对象
	// 这是缓存复用的核心方法：将预分配的缓存内存和元数据重新设置到目标 Vector 中
	// 避免了重新分配内存，大大提高了性能
	// @param result: 要重置的目标 Vector
	// @param buffer: 当前缓存缓冲区的指针（传入自身以维持引用计数）
	void ResetFromCache(Vector &result, const buffer_ptr<VectorBuffer> &buffer) {
		// 断言：确保缓存的类型与目标 Vector 的类型一致
		D_ASSERT(type == result.GetType());
		// 获取物理类型
		auto internal_type = type.InternalType();
		// 将向量类型设置为 FLAT_VECTOR（扁平向量，最常用的向量类型）
		result.vector_type = VectorType::FLAT_VECTOR;
		// 将缓存缓冲区赋值给结果向量（使用共享指针，增加引用计数）
		AssignSharedPointer(result.buffer, buffer);
		// 重置有效性位掩码（validity mask），用于标记哪些元素是 NULL
		result.validity.Reset(capacity);
		// 根据不同的物理类型，设置不同的数据结构
		switch (internal_type) {
		case PhysicalType::LIST: {
			// LIST 类型的重置
			// 将偏移量数组指针赋值给 result.data
			result.data = owned_data.get();
			// 重新初始化 VectorListBuffer，赋值辅助缓冲区
			AssignSharedPointer(result.auxiliary, auxiliary);
			// 递归处理子元素：获取子缓存
			auto &child_cache = child_caches[0]->Cast<VectorCacheBuffer>();
			// 获取列表缓冲区
			auto &list_buffer = result.auxiliary->Cast<VectorListBuffer>();
			// 设置子元素的容量
			list_buffer.SetCapacity(child_cache.capacity);
			// 重置列表大小为 0（空列表）
			list_buffer.SetSize(0);
			// 清空辅助数据
			list_buffer.SetAuxiliaryData(nullptr);

			// 获取子向量并递归重置
			auto &list_child = list_buffer.GetChild();
			child_cache.ResetFromCache(list_child, child_caches[0]);
			break;
		}
		case PhysicalType::ARRAY: {
			// ARRAY 类型的重置
			// 固定大小数组没有自己的 data（偏移量），所有数据都在子向量中
			result.data = nullptr;
			// 重新初始化 VectorArrayBuffer
			// auxiliary->SetAuxiliaryData(nullptr);  // 注释掉的代码
			AssignSharedPointer(result.auxiliary, auxiliary);

			// 递归处理子元素
			auto &child_cache = child_caches[0]->Cast<VectorCacheBuffer>();
			// 获取数组的子向量
			auto &array_child = result.auxiliary->Cast<VectorArrayBuffer>().GetChild();
			// 递归重置子向量
			child_cache.ResetFromCache(array_child, child_caches[0]);
			break;
		}
		case PhysicalType::STRUCT: {
			// STRUCT 类型的重置
			// 结构体没有自己的 data（所有数据都在子字段中）
			result.data = nullptr;
			// 重新初始化 VectorStructBuffer
			auxiliary->SetAuxiliaryData(nullptr);
			AssignSharedPointer(result.auxiliary, auxiliary);
			// 递归处理所有子字段
			auto &children = result.auxiliary->Cast<VectorStructBuffer>().GetChildren();
			for (idx_t i = 0; i < children.size(); i++) {
				// 获取第 i 个子字段的缓存
				auto &child_cache = child_caches[i]->Cast<VectorCacheBuffer>();
				// 递归重置第 i 个子字段
				child_cache.ResetFromCache(*children[i], child_caches[i]);
			}
			break;
		}
		default:
			// 基础类型的重置：简单类型（INT, DOUBLE, VARCHAR 等）
			// 直接将缓存的数据指针赋值给 result.data
			result.data = owned_data.get();
			// 基础类型没有辅助数据，重置为空
			result.auxiliary.reset();
			break;
		}
	}

	// GetAllocator: 获取内存分配器
	// 返回用于分配 owned_data 的分配器
	optional_ptr<Allocator> GetAllocator() const override {
		return owned_data.GetAllocator();
	}

	// GetType: 获取缓存的逻辑类型
	const LogicalType &GetType() {
		return type;
	}

private:
	//! 向量缓存的逻辑类型（如 INTEGER, VARCHAR, LIST<INTEGER> 等）
	LogicalType type;
	//! 拥有的数据内存块
	//! 对于基础类型：存储实际的数据值
	//! 对于 LIST：存储偏移量数组
	//! 对于 STRUCT/ARRAY：为空（数据在子缓存中）
	AllocatedData owned_data;
	//! 子缓存列表（用于嵌套类型）
	//! LIST: 包含一个子缓存（存储列表元素）
	//! ARRAY: 包含一个子缓存（存储数组元素）
	//! STRUCT: 包含多个子缓存（每个字段一个）
	vector<buffer_ptr<VectorBuffer>> child_caches;
	//! 辅助数据缓冲区（用于复杂类型）
	//! 存储 VectorListBuffer、VectorArrayBuffer 或 VectorStructBuffer
	buffer_ptr<VectorBuffer> auxiliary;
	//! 向量的容量（可以存储的最大元素数量）
	idx_t capacity;
};

// VectorCache 类的实现
// VectorCache 是对外的接口类，内部持有一个 VectorCacheBuffer

// 默认构造函数：创建一个空的缓存（buffer 为空指针）
VectorCache::VectorCache() : buffer(nullptr) {
}

// 带参数的构造函数：创建指定类型和容量的缓存
// @param allocator: 内存分配器
// @param type_p: 向量的逻辑类型
// @param capacity_p: 向量容量
VectorCache::VectorCache(Allocator &allocator, const LogicalType &type_p, const idx_t capacity_p) {
	// 创建 VectorCacheBuffer 对象
	buffer = make_buffer<VectorCacheBuffer>(allocator, type_p, capacity_p);
}

// ResetFromCache: 从缓存重置 Vector
// 这是 VectorCache 的核心功能：将缓存的内存和元数据应用到目标 Vector
// 使得该 Vector 可以复用预分配的内存，避免频繁的分配/释放操作
// @param result: 要重置的目标 Vector
void VectorCache::ResetFromCache(Vector &result) const {
	// 如果缓存为空，直接返回（不做任何操作）
	if (!buffer) {
		return;
	}
	// 将 buffer 转换为 VectorCacheBuffer 类型
	auto &vector_cache = buffer->Cast<VectorCacheBuffer>();
	// 调用 VectorCacheBuffer 的 ResetFromCache 方法进行实际的重置操作
	vector_cache.ResetFromCache(result, buffer);
}

// GetType: 获取缓存的逻辑类型
// @return 缓存的逻辑类型引用
const LogicalType &VectorCache::GetType() const {
	// 断言缓存非空
	D_ASSERT(buffer);
	// 将 buffer 转换为 VectorCacheBuffer 类型
	auto &vector_cache = buffer->Cast<VectorCacheBuffer>();
	// 返回类型
	return vector_cache.GetType();
}

} // namespace duckdb
