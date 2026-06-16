// ============================================================
// MODULE : hybrid_hghv_cordic_24b  —  log2(Q)  Vectoring Mode
// FORMAT : Q4.20  ONE=1048576  SCALE=2^20  range (-8,+8)
// SCHEDULE: R8×4 + R4×2  (6 CORDIC stages)
// INPUT  : Q_fixed [23:0] in Q4.20, Q in [0.5, 3.0]
// OUTPUT : log2_Q_fixed [23:0] in Q4.20
// LATENCY: 8 clk  (stage0 + 6 CORDIC + out_reg)
// ============================================================
`timescale 1ns/1ps

module hybrid_hghv_cordic (
    input  wire        clk, rst_n, valid_in,
    input  wire [23:0] Q_fixed,
    output reg         valid_out,
    output reg  [23:0] log2_Q_fixed
);
    localparam signed [23:0] ONE = 24'sd1048576; // 1.0 in Q4.20

    // ── Angle LUT: 9 stored constants (Scaled to Q4.20) ───────────────
    localparam signed [23:0] S1A1 = 24'sd190091;   // atanh(1/8)
    localparam signed [23:0] S1A2 = 24'sd386382;   // atanh(2/8)
    localparam signed [23:0] S1A3 = 24'sd596379;   // atanh(3/8)
    localparam signed [23:0] S1A4 = 24'sd830977;   // atanh(4/8)
    localparam signed [23:0] S2A1 = 24'sd23639;    // atanh(1/64)
    localparam signed [23:0] S2A2 = 24'sd47289;    // atanh(2/64)
    localparam signed [23:0] S2A3 = 24'sd70963;    // atanh(3/64)
    localparam signed [23:0] S2A4 = 24'sd94672;    // atanh(4/64)
    localparam signed [23:0] ANC  = 24'sd2955;     // stage-3 anchor

    reg signed [23:0] x0,y0,z0; reg v0;
    reg signed [23:0] x1,y1,z1; reg v1;
    reg signed [23:0] x2,y2,z2; reg v2;
    reg signed [23:0] x3,y3,z3; reg v3;
    reg signed [23:0] x4,y4,z4; reg v4;
    reg signed [23:0] x5,y5,z5; reg v5;
    reg signed [23:0] x6,y6,z6; reg v6;

    // Stage 0
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin x0<=0;y0<=0;z0<=0;v0<=0; end
        else begin
            x0<=$signed(Q_fixed)+ONE; y0<=$signed(Q_fixed)-ONE;
            z0<=24'sd0; v0<=valid_in;
        end
    end

    // ── DIGIT COMPARATOR MACRO ───────────────────────────────────────
    // For each stage, the 27-bit pattern is:
    //   ya = |y| zero-extended to 48 bits, shifted, saturated to 27b
    //   x27 = x zero-extended to 27 bits (Xmax ~ 4.5, 7*Xmax fits in 26 bits)
    //   x7=x27*7, x5=x27*5, x3=x27*3  via shift-add (no multiplier)
    //   da  = priority compare
    
    // ── Stage 1  S=3  dmax=4 ────────────────────────────────────────
    wire [23:0] s1_ya = y0[23] ? ~y0+1'b1 : y0;
    wire [26:0] s1_x  = {3'b000, x0};              
    wire [26:0] s1_x7 = (s1_x<<<3) - s1_x;
    wire [26:0] s1_x5 = (s1_x<<<2) + s1_x;
    wire [26:0] s1_x3 = (s1_x<<<1) + s1_x;
    
    wire [47:0] s1_full_lhs = {24'd0, s1_ya} << 4; 
    wire        s1_ovf      = |s1_full_lhs[47:27];
    wire [26:0] s1_lhs      = s1_ovf ? 27'h7FFFFFF : s1_full_lhs[26:0];
    
    wire [2:0]  s1_da  = (s1_lhs>=s1_x7)?3'd4:(s1_lhs>=s1_x5)?3'd3:
                         (s1_lhs>=s1_x3)?3'd2:(s1_lhs>=s1_x)?3'd1:3'd0;
                         
    wire signed [23:0] s1_di  = y0[23] ? (-{{21{1'b0}},s1_da}) : ({{21{1'b0}},s1_da});
    wire signed [23:0] s1_ap  = (s1_da==3'd4)?S1A4:(s1_da==3'd3)?S1A3:
                                (s1_da==3'd2)?S1A2:(s1_da==3'd1)?S1A1:24'sd0;
    wire signed [23:0] s1_ang = s1_di[23] ? -s1_ap : s1_ap;
    
    wire signed [47:0] s1_py  = $signed(s1_di)*$signed({{24{y0[23]}},y0});
    wire signed [47:0] s1_px  = $signed(s1_di)*$signed({{24{x0[23]}},x0});
    wire signed [23:0] s1_xn  = x0 - s1_py[26:3];
    wire signed [23:0] s1_yn  = y0 - s1_px[26:3];
    wire signed [23:0] s1_zn  = z0 + s1_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x1<=0;y1<=0;z1<=0;v1<=0; end
        else begin x1<=s1_xn;y1<=s1_yn;z1<=s1_zn;v1<=v0; end

    // ── Stage 2  S=6  dmax=4 ────────────────────────────────────────
    wire [23:0] s2_ya = y1[23] ? ~y1+1'b1 : y1;
    wire [26:0] s2_x  = {3'b000, x1};
    wire [26:0] s2_x7 = (s2_x<<<3) - s2_x;
    wire [26:0] s2_x5 = (s2_x<<<2) + s2_x;
    wire [26:0] s2_x3 = (s2_x<<<1) + s2_x;
    
    wire [47:0] s2_full_lhs = {24'd0, s2_ya} << 7;
    wire        s2_ovf      = |s2_full_lhs[47:27];
    wire [26:0] s2_lhs      = s2_ovf ? 27'h7FFFFFF : s2_full_lhs[26:0];
    
    wire [2:0]  s2_da  = (s2_lhs>=s2_x7)?3'd4:(s2_lhs>=s2_x5)?3'd3:
                         (s2_lhs>=s2_x3)?3'd2:(s2_lhs>=s2_x)?3'd1:3'd0;
                         
    wire signed [23:0] s2_di  = y1[23] ? (-{{21{1'b0}},s2_da}) : ({{21{1'b0}},s2_da});
    wire signed [23:0] s2_ap  = (s2_da==3'd4)?S2A4:(s2_da==3'd3)?S2A3:
                                (s2_da==3'd2)?S2A2:(s2_da==3'd1)?S2A1:24'sd0;
    wire signed [23:0] s2_ang = s2_di[23] ? -s2_ap : s2_ap;
    
    wire signed [47:0] s2_py  = $signed(s2_di)*$signed({{24{y1[23]}},y1});
    wire signed [47:0] s2_px  = $signed(s2_di)*$signed({{24{x1[23]}},x1});
    wire signed [23:0] s2_xn  = x1 - s2_py[29:6];
    wire signed [23:0] s2_yn  = y1 - s2_px[29:6];
    wire signed [23:0] s2_zn  = z1 + s2_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x2<=0;y2<=0;z2<=0;v2<=0; end
        else begin x2<=s2_xn;y2<=s2_yn;z2<=s2_zn;v2<=v1; end

    // ── Stage 3  S=9  dmax=4  angle: ANC + shift-add ────────────────
    wire [23:0] s3_ya = y2[23] ? ~y2+1'b1 : y2;
    wire [26:0] s3_x  = {3'b000, x2};
    wire [26:0] s3_x7 = (s3_x<<<3) - s3_x;
    wire [26:0] s3_x5 = (s3_x<<<2) + s3_x;
    wire [26:0] s3_x3 = (s3_x<<<1) + s3_x;
    
    wire [47:0] s3_full_lhs = {24'd0, s3_ya} << 10;
    wire        s3_ovf      = |s3_full_lhs[47:27];
    wire [26:0] s3_lhs      = s3_ovf ? 27'h7FFFFFF : s3_full_lhs[26:0];
    
    wire [2:0]  s3_da  = (s3_lhs>=s3_x7)?3'd4:(s3_lhs>=s3_x5)?3'd3:
                         (s3_lhs>=s3_x3)?3'd2:(s3_lhs>=s3_x)?3'd1:3'd0;
                         
    wire signed [23:0] s3_di  = y2[23] ? (-{{21{1'b0}},s3_da}) : ({{21{1'b0}},s3_da});
    wire signed [23:0] s3_ap  = (s3_da==3'd4)?(ANC<<<2):
                                (s3_da==3'd3)?(ANC+(ANC<<<1)):
                                (s3_da==3'd2)?(ANC<<<1):
                                (s3_da==3'd1)? ANC:24'sd0;
    wire signed [23:0] s3_ang = s3_di[23] ? -s3_ap : s3_ap;
    
    wire signed [47:0] s3_py  = $signed(s3_di)*$signed({{24{y2[23]}},y2});
    wire signed [47:0] s3_px  = $signed(s3_di)*$signed({{24{x2[23]}},x2});
    wire signed [23:0] s3_xn  = x2 - s3_py[32:9];
    wire signed [23:0] s3_yn  = y2 - s3_px[32:9];
    wire signed [23:0] s3_zn  = z2 + s3_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x3<=0;y3<=0;z3<=0;v3<=0; end
        else begin x3<=s3_xn;y3<=s3_yn;z3<=s3_zn;v3<=v2; end

    // ── Stage 4  S=12  dmax=4  base=ANC>>3 ──────────────────────────
    wire signed [23:0] s4_b   = ANC >>> 3;
    wire [23:0] s4_ya = y3[23] ? ~y3+1'b1 : y3;
    wire [26:0] s4_x  = {3'b000, x3};
    wire [26:0] s4_x7 = (s4_x<<<3) - s4_x;
    wire [26:0] s4_x5 = (s4_x<<<2) + s4_x;
    wire [26:0] s4_x3 = (s4_x<<<1) + s4_x;
    
    wire [47:0] s4_full_lhs = {24'd0, s4_ya} << 13;
    wire        s4_ovf      = |s4_full_lhs[47:27];
    wire [26:0] s4_lhs      = s4_ovf ? 27'h7FFFFFF : s4_full_lhs[26:0];
    
    wire [2:0]  s4_da  = (s4_lhs>=s4_x7)?3'd4:(s4_lhs>=s4_x5)?3'd3:
                         (s4_lhs>=s4_x3)?3'd2:(s4_lhs>=s4_x)?3'd1:3'd0;
                         
    wire signed [23:0] s4_di  = y3[23] ? (-{{21{1'b0}},s4_da}) : ({{21{1'b0}},s4_da});
    wire signed [23:0] s4_ap  = (s4_da==3'd4)?(s4_b<<<2):
                                (s4_da==3'd3)?(s4_b+(s4_b<<<1)):
                                (s4_da==3'd2)?(s4_b<<<1):
                                (s4_da==3'd1)? s4_b:24'sd0;
    wire signed [23:0] s4_ang = s4_di[23] ? -s4_ap : s4_ap;
    
    wire signed [47:0] s4_py  = $signed(s4_di)*$signed({{24{y3[23]}},y3});
    wire signed [47:0] s4_px  = $signed(s4_di)*$signed({{24{x3[23]}},x3});
    wire signed [23:0] s4_xn  = x3 - s4_py[35:12];
    wire signed [23:0] s4_yn  = y3 - s4_px[35:12];
    wire signed [23:0] s4_zn  = z3 + s4_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x4<=0;y4<=0;z4<=0;v4<=0; end
        else begin x4<=s4_xn;y4<=s4_yn;z4<=s4_zn;v4<=v3; end

    // ── Stage 5  S=14  dmax=2  base=ANC>>5 ──────────────────────────
    wire signed [23:0] s5_b   = ANC >>> 5;
    wire [23:0] s5_ya = y4[23] ? ~y4+1'b1 : y4;
    wire [26:0] s5_x  = {3'b000, x4};
    wire [26:0] s5_x3 = (s5_x<<<1) + s5_x;
    
    wire [47:0] s5_full_lhs = {24'd0, s5_ya} << 15;
    wire        s5_ovf      = |s5_full_lhs[47:27];
    wire [26:0] s5_lhs      = s5_ovf ? 27'h7FFFFFF : s5_full_lhs[26:0];
    
    wire [2:0]  s5_da  = (s5_lhs>=s5_x3)?3'd2:(s5_lhs>=s5_x)?3'd1:3'd0;
    wire signed [23:0] s5_di  = y4[23] ? (-{{21{1'b0}},s5_da}) : ({{21{1'b0}},s5_da});
    wire signed [23:0] s5_ap  = (s5_da==3'd2)?(s5_b<<<1):
                                (s5_da==3'd1)? s5_b:24'sd0;
    wire signed [23:0] s5_ang = s5_di[23] ? -s5_ap : s5_ap;
    
    wire signed [47:0] s5_py  = $signed(s5_di)*$signed({{24{y4[23]}},y4});
    wire signed [47:0] s5_px  = $signed(s5_di)*$signed({{24{x4[23]}},x4});
    wire signed [23:0] s5_xn  = x4 - s5_py[37:14];
    wire signed [23:0] s5_yn  = y4 - s5_px[37:14];
    wire signed [23:0] s5_zn  = z4 + s5_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x5<=0;y5<=0;z5<=0;v5<=0; end
        else begin x5<=s5_xn;y5<=s5_yn;z5<=s5_zn;v5<=v4; end

    // ── Stage 6  S=16  dmax=2  base=ANC>>7 ──────────────────────────
    wire signed [23:0] s6_b   = ANC >>> 7;
    wire [23:0] s6_ya = y5[23] ? ~y5+1'b1 : y5;
    wire [26:0] s6_x  = {3'b000, x5};
    wire [26:0] s6_x3 = (s6_x<<<1) + s6_x;
    
    wire [47:0] s6_full_lhs = {24'd0, s6_ya} << 17;
    wire        s6_ovf      = |s6_full_lhs[47:27];
    wire [26:0] s6_lhs      = s6_ovf ? 27'h7FFFFFF : s6_full_lhs[26:0];
    
    wire [2:0]  s6_da  = (s6_lhs>=s6_x3)?3'd2:(s6_lhs>=s6_x)?3'd1:3'd0;
    wire signed [23:0] s6_di  = y5[23] ? (-{{21{1'b0}},s6_da}) : ({{21{1'b0}},s6_da});
    wire signed [23:0] s6_ap  = (s6_da==3'd2)?(s6_b<<<1):
                                (s6_da==3'd1)? s6_b:24'sd0;
    wire signed [23:0] s6_ang = s6_di[23] ? -s6_ap : s6_ap;
    
    wire signed [47:0] s6_py  = $signed(s6_di)*$signed({{24{y5[23]}},y5});
    wire signed [47:0] s6_px  = $signed(s6_di)*$signed({{24{x5[23]}},x5});
    wire signed [23:0] s6_xn  = x5 - s6_py[39:16];
    wire signed [23:0] s6_yn  = y5 - s6_px[39:16];
    wire signed [23:0] s6_zn  = z5 + s6_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x6<=0;y6<=0;z6<=0;v6<=0; end
        else begin x6<=s6_xn;y6<=s6_yn;z6<=s6_zn;v6<=v5; end

    // Output: log2(Q) = z6 * 2
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin valid_out<=0; log2_Q_fixed<=0; end
        else begin valid_out<=v6; log2_Q_fixed<=z6<<<1; end

endmodule
