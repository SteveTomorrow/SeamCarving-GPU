# SeamCarving-GPU

khắc đường may

Đây là một triển khai C++ của thuật toán khắc đường may. Mục đích của thuật toán khắc đường nối là chỉnh lại kích thước hình ảnh mà không làm biến dạng các phần "quan trọng" của hình ảnh, giống như trường hợp bạn chỉ cố gắng thay đổi kích thước hình ảnh thông thường.

Thuật toán khắc đường nối tính toán đường nối (đường dẫn gồm 8 pixel được kết nối) từ trên xuống dưới hoặc từ trái sang phải. Một đường nối được tính toán bằng cách duyệt qua bản đồ năng lượng tích lũy của hình ảnh và chọn đường đi có chi phí thấp nhất. Để tạo bản đồ năng lượng tích lũy, chúng ta cần một hình ảnh năng lượng. Để có được hình ảnh năng lượng, chúng tôi sử dụng một gradient theo hướng x và y của hình ảnh, sau đó kết hợp chúng để tạo thành hình ảnh năng lượng: ...

Từ hình ảnh năng lượng này, chúng ta có thể tính toán bản đồ năng lượng tích lũy. Chúng tôi di chuyển theo một hướng (trong ví dụ này là từ trên xuống dưới). Tại mỗi hàng, chúng tôi lặp qua tất cả các pixel. Tại mỗi pixel, chúng tôi kiểm tra ba pixel phía trên phía trên pixel hiện tại, chọn mức tối thiểu trong số đó và thêm chúng vào tổng số đang chạy tại pixel đó. Cuối cùng, chúng tôi tạo thành một hình ảnh có thể được biểu thị dưới dạng: ... (màu xanh biểu thị năng lượng thấp, màu đỏ biểu thị năng lượng cao)

Từ đây, chúng tôi nhìn vào hàng cuối cùng của bản đồ, chọn giá trị tối thiểu và theo dõi đường dẫn tối thiểu cho đến khi đạt đến đỉnh của hình ảnh. Điều này trở thành đường may mà chúng tôi loại bỏ. Rửa sạch và lặp lại.
