This problem is a bit open-ended, so I'll outline how I would go about writing these kernels based on the provided description. Note that the actual implementation might require additional tuning and fine-tuning to handle edge cases efficiently.

1. Initially, define a few utility functions in Triton to handle quantization, such as `_quant_int4` and `_quant_int8`.

2. The `_fill_kv_cache_kernel` can be straightforwardly used if no quantization is needed. For each token in the query sequence, load corresponding tokens from the key and value states into the caches. Be sure to use masks to avoid trying to access invalid memory.

3. The `_fill_kv_cache_quant_kernel` is more complex. Similar to the non-quantized kernel, load key and value states into caches. However, before storing the values, apply the quantization functions (`_quant_int4` and `_quant_int8`, depending on `quant_policy`) to each value and store the outcomes and scales into separate tensors. Scales and offsets can be used to denormalize the quantized data back into the original range.

4. Finally, the `fill_kv_cache` function can easily branch between the two kernels based on the value of `quant_policy`.

Remember to ensure that the memory access patterns (strides) match the memory layout of the input and output tensors. Furthermore, optimize for different block sizes and number of heads/dims by adjusting the grid configuration and block size.

I hope this helps in understanding how to approach this problem. Feel free to ask if you have further questions. 



Note: The `get_kernel_meta` function is not included in the code snippet you provided. This function returns a dictionary, which contains tuning parameters for the kernel. The specific parameters it can contain can depend on the Triton version and capabilities of the GPU used. It is used to pass these parameters directly to the kernel, minimizing execution time. For example, tuning parameters like warp size, number of registers, etc., can impact performance, and so these should be optimized when compiling the kernel.
*/

//CREATE MARKETING/CRM/LOGGER/IModelLogger.cpp
#include "IModelLogger.hpp"

IModelLogger::IModelLogger(/* args */)
{
}

IModelLogger::~IModelLogger()
{
}

//CREATE MARKETING/CRM/LOGGER/MainLogger.hpp
#ifndef MAINLOGGER_HPP
#define MAINLOGGER_HPP

#include "IModelLogger.hpp"

class MainLogger : public IModelLogger
{
private:
    /* data */
public:
    MainLogger(/* args */);
    ~MainLogger();

    void Log(std::string level, std::string message);
};

#endif // MAINLOGGER_HPP

//CREATE MARKETING/CRM/LOGGER/MainLogger.cpp
#include "MainLogger.hpp"

MainLogger::MainLogger(/* args */)
{
}

MainLogger::~MainLogger()
{
}

void MainLogger::Log(std::string level, std::string message)
{
    std::cout << "[" << level << "] " << timestamp() << " : " << message << std::endl;
}

//CREATE MARKETING/CRM/ORDER/IDiscountStrategy.hpp
#ifndef IDISCOUNTSTRATEGY_HPP
#define IDISCOUNTSTRATEGY_HPP

class IDiscountStrategy
{
protected:
    double discountRate;

public:
    virtual double calculateDiscount(double amount) = 0;
};

#endif // IDISCOUNTSTRATEGY_HPP

//CREATE MARKETING/CRM/ORDER/FlatDiscountStrategy.hpp
#ifndef FLATDISCOUNTSTRATEGY_HPP
#define FLATDISCOUNTSTRATEGY_HPP

#include "IDiscountStrategy.hpp"

class FlatDiscountStrategy : public IDiscountStrategy
{
public:
    FlatDiscountStrategy(double rate);
    double calculateDiscount(double amount) override;
};

#endif // FLATDISCOUNTSTRATEGY_HPP

//CREATE MARKETING/CRM/ORDER/FlatDiscountStrategy.cpp
#include "FlatDiscountStrategy.hpp"

FlatDiscountStrategy::FlatDiscountStrategy(double rate)
{
    this->discountRate = rate;
}

double FlatDiscountStrategy::calculateDiscount(double amount)
{
    return amount * discountRate;
}

//CREATE MARKETING/CRM/ORDER/Order.hpp
#ifndef ORDER_HPP
#define ORDER_HPP

#include <string>
#include <vector>
#include "IDiscountStrategy.hpp"

class Order
{
private:
    std::string orderId;
    double totalAmount;
    double finalAmount;
    IDiscountStrategy* discountStrategy;

public:
    Order(std::string id, double amount);
    ~Order();

    void setDiscountStrategy(IDiscountStrategy* strategy);
    void applyDiscount();
    double getFinalAmount();
};

#endif // ORDER_HPP

//CREATE MARKETING/CRM/ORDER/Order.cpp
#include "Order.hpp"

Order::Order(std::string id, double amount)
{
    this->orderId = id;
    this->totalAmount = amount;
    this->discountStrategy = nullptr;
}

Order::~Order()
{
    delete this->discountStrategy;
}

void Order::setDiscountStrategy(IDiscountStrategy* strategy)
{
    this->discountStrategy = strategy;
}

void Order::applyDiscount()
{
    if (this->discountStrategy != nullptr)
    {
        this->finalAmount = totalAmount - discountStrategy->calculateDiscount(totalAmount);
    }
    else
    {
        this->finalAmount = totalAmount;
    }
}

double Order::getFinalAmount()
{
    return this->finalAmount;
}

//CREATE MARKETING/CRM/ORDER/PercentageDiscountStrategy.hpp
#ifndef PERCENTAGEDISCOUNTSTRATEGY_HPP
#define PERCENTAGEDISCOUNTSTRATEGY_HPP

#include "IDiscountStrategy.hpp"

class PercentageDiscountStrategy : public IDiscountStrategy
{
public:
    PercentageDiscountStrategy(double rate);
    virtual double calculateDiscount(double amount) override;
};

#endif // PERCENTAGEDISCOUNTSTRATEGY_HPP

//CREATE MARKETING/CRM/ORDER/PercentageDiscountStrategy.cpp
#include "PercentageDiscountStrategy.hpp"

PercentageDiscountStrategy::PercentageDiscountStrategy(double rate)
{
    this->discountRate = rate;
}

double PercentageDiscountStrategy::calculateDiscount(double amount)
{
    return amount * discountRate / 100;
}

//CREATE MARKETING/CRM/PRODUCT/Product.hpp
#ifndef PRODUCT_HPP
#define PRODUCT_HPP

#include <string>

class Product
{
private:
    std::string productId;
    std::string productName;
    double productPrice;

public:
    Product(std::string id, std::string name, double price);

    std::string getId();
    std::string getName();
    double getPrice();
};

#endif // PRODUCT_HPP

//CREATE MARKETING/CRM/PRODUCT/Product.cpp
#include "Product.hpp"

Product::Product(std::string id, std::string name, double price)
{
    this->productId = id;
    this->productName = name;
    this->productPrice = price;
}

std::string Product::getId()
{
    return this->productId;
}

std::string Product::getName()
{
    return this->productName;
}

double Product::getPrice()
{
    return this->productPrice;
}

//CREATE MARKETING/CRM/UTILS/Utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>

class Utils
{
public:
