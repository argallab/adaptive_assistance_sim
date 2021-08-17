#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h> //tf::matrixTFToEigen

// #include "mico_interaction/getJacobian.h"

#include <moveit/move_group_interface/move_group.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>

#include <jaco_teleop/dlib/optimization.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>

#include <cmath>
// constants for optimization
#define DELTA_JOINT_PENALTY 0.5
#define LAMBDA_DQDIST 0.01
#define JOINT_LIMIT_TOLERANCE 0.03


typedef dlib::matrix<double> dlib_matr; //runtime sized 6xn matrix
typedef dlib::matrix<double, 0, 1> dlib_vec; //runtime sized nx1 vector

//////    header  ////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
class TeleopClass
{
public:
  TeleopClass();
  ~TeleopClass();

  void UpdateAll();
  void PublishJointVelocities();

  dlib_vec dq_target_; //targeted vel that takes us away from joint pos limits
  dlib_vec current_twist_6_;

  dlib_vec dq_min_current_; //desired joint vel limit for optimization
  dlib_vec dq_max_current_;
  dlib_vec dq_opt_; //dq_opt_ will be used in the optimization as a reference and contains the optimal joint velocities

  int dof_; //degrees of freedom

  bool update_flag_; //flag is set to true if twist and joint_state callbacks are not empty anymore

  dlib_matr jacobian_;

  std::string control_mode_;

private:
  void TwistCB(const std_msgs::Float32MultiArray::ConstPtr& new_twist);
  void JointStateCB(const sensor_msgs::JointState::ConstPtr& new_joint_state);

  void UpdateTwist();
  void FilterTwist();
  void ChangeTwistOrientationReference();
  void UpdateCurrentVelocityLimits();
  void UpdateJacobian();
  void UpdateTargetVelocity();
  void UpdateJointStates();

  int num_fingers_; // number of controllable fingers

////// ROS stuff //////////
  ros::NodeHandle nh_;

  tf::TransformListener tf_listener_;
  tf::StampedTransform tf_;
  dlib_matr dlib_w_to_eef_;
  Eigen::Matrix3d Eigen_w_to_eef_;


//////  twist variables   ///////
  std_msgs::Float32MultiArray::ConstPtr current_control_input_ptr_;
  ros::Subscriber twist_sub_;

  std::vector <dlib_vec> twist_6_filter_list_;

//////  jointState  ///////

  sensor_msgs::JointState::ConstPtr current_joint_state_ptr_;
  ros::Subscriber joint_state_sub_;

/////   moveit variables ///////////
  robot_state::RobotStatePtr robot_state_ptr_;
  // moveit::planning_interface::MoveGroup *  move_group_ptr_;
  const robot_state::JointModelGroup * arm_joint_model_group_;
  const robot_state::JointModelGroup * gripper_joint_model_group_;
  //moveit return the jacobian as eigen matrix
  Eigen::MatrixXd jacobian_Eigen_;
  // reference for querying the jacobian through moveit
  Eigen::Vector3d jacobian_reference_;
  const robot_state::LinkModel* hand_link_model_;


  dlib_vec q_;
  dlib_vec q_max_; //hardware joint pos limit
  dlib_vec q_min_;

  dlib_vec dq_;
  dlib_vec dq_min_; //hardware joint vel limit
  dlib_vec dq_max_;

  dlib_vec qdiff_lower_; //variable to compute dq_target
  dlib_vec qdiff_upper_;//variable to compute dq_target

  dlib_vec ones_; //column vector with ones
  dlib_vec zeros_;

  dlib_vec current_input_finger_;

  ros::Publisher vel_j0_pub_;
  ros::Publisher vel_j1_pub_;
  ros::Publisher vel_j2_pub_;
  ros::Publisher vel_j3_pub_;
  ros::Publisher vel_j4_pub_;
  ros::Publisher vel_j5_pub_;
  ros::Publisher finger_1_pub_;
  ros::Publisher finger_2_pub_;

  std_msgs::Float64 vel0_, vel1_, vel2_, vel3_, vel4_, vel5_, fin1_, fin2_;
};
